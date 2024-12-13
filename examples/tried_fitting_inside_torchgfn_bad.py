# torchgfn_example.py
# Ensure you have the GFlowNet repository and dependencies installed.

from collections import deque
from copy import deepcopy
from typing import Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from gfn.env import Env
from gfn.states import States
from gfn.actions import Actions
from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP


class RedistrictEnv(Env):
    """
    Environment for assigning nodes in a 3x3 grid to districts.

    Precincts: 0 through 8 arranged as a 3x3 grid:
        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
        |    |    |
        6 -- 7 -- 8

    Populations = [30,20,10,20,10,20,30,10,0] sum=150
    num_districts=3, ideal_pop=50 each

    Goal: Assign nodes to 3 districts so that each district is contiguous and has population ~50.

    Reward = 1/(1+pop_deviation+contiguity_penalty)
    pop_deviation = mean(|dist_pop - 50|/50)
    contiguity_penalty = number_of_noncontiguous_districts
    """

    def __init__(self, num_districts=3, device: Optional[torch.device] = None):
        self.graph = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        # Updated populations to include node 8 with population 0
        self.device = device
        self.populations = {
            0: 30, 1: 20, 2: 5,
            3: 20, 4: 10, 5: 20,
            6: 30, 7: 10, 8: 5
        }
        self.num_nodes = 9
        self.num_districts = num_districts
        self.nodes = list(self.graph.keys())
        # Actions: num_nodes * num_districts + 1 (exit action)
        self.n_actions = self.num_nodes * self.num_districts + 1

        total_pop = sum(self.populations.values())  # should be 150
        self.ideal_pop = total_pop / self.num_districts

        # Define the initial state (s0)
        s0 = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)

        # Define the state shape and action shape
        state_shape = (self.num_nodes,)  # Each node has an assigned district or -1
        action_shape = (2,)  # Actions are (node_id, district_id) or exit

        # Define the dummy and exit actions
        dummy_action = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
        exit_action = torch.tensor([self.num_nodes * self.num_districts], dtype=torch.long, device=self.device)

        # Call the parent class constructor
        super().__init__(
            s0=s0,
            state_shape=state_shape,
            action_shape=action_shape,
            dummy_action=dummy_action,
            exit_action=exit_action,
            device_str=device
        )

    def reset(self, batch_shape: Tuple[int, ...]) -> States:
        """
        Reset the environment for a batch of states.
        batch_shape: (n,) where n is the number of trajectories to start.

        Returns a States object with shape (n, num_nodes).
        """
        batch_size = batch_shape[0]
        assignments = self.s0.expand(batch_size, -1)
        # Remove is_initial_state and is_sink_state arguments
        return States(tensor=assignments)

    def actions_from_tensor(self, actions_tensor: torch.Tensor) -> Actions:
        batch_size = actions_tensor.shape[0]
        actions = torch.empty((batch_size, 2), dtype=torch.long, device=self.device)
        exit_mask = (actions_tensor == self.num_nodes * self.num_districts)

        normal_actions = actions_tensor[~exit_mask]
        node_ids = normal_actions // self.num_districts
        dist_ids = normal_actions % self.num_districts

        # Fill normal actions
        actions[~exit_mask, 0] = node_ids
        actions[~exit_mask, 1] = dist_ids

        # Fill exit actions with (-1, -1)
        actions[exit_mask] = torch.tensor([-1, -1], device=self.device)

        return Actions(tensor=actions)

    def actions_from_batch_shape(self, batch_shape: Tuple[int, ...]) -> Actions:
        # Create a dummy Actions object
        batch_size = batch_shape[0]
        return Actions(tensor=torch.full((batch_size, 2), -1, dtype=torch.long, device=self.device))

    def is_action_valid(self, states: States, actions: Actions) -> torch.Tensor:
        assignments = states.tensor
        node_ids = actions.tensor[:, 0]

        exit_mask = (node_ids == -1)
        valid = torch.zeros_like(exit_mask, dtype=torch.bool, device=self.device)
        # Exit action always valid
        valid[exit_mask] = True

        # For normal actions, the chosen node must be unassigned
        non_exit_mask = ~exit_mask
        if non_exit_mask.any():
            node_ids_non_exit = node_ids[non_exit_mask]
            node_not_assigned = (assignments[non_exit_mask, node_ids_non_exit] == -1)
            valid[non_exit_mask] = node_not_assigned

        return valid

    def step(self, states: States, actions: Actions) -> States:
        old_assignments = states.tensor.clone()
        batch_size = old_assignments.shape[0]

        node_ids = actions.tensor[:, 0]
        dist_ids = actions.tensor[:, 1]

        # Assign if not exit
        non_exit_mask = (node_ids != -1)
        old_assignments[non_exit_mask, node_ids[non_exit_mask]] = dist_ids[non_exit_mask]

        # Just return a new States with updated assignments
        return States(tensor=old_assignments)

    def backward_step(self, states: States, actions: Actions) -> States:
        old_assignments = states.tensor.clone()
        batch_size = old_assignments.shape[0]

        node_ids = actions.tensor[:, 0]

        if (node_ids == -1).any():
            raise NotImplementedError("Backward step from exit action is not defined.")

        # Revert assignment
        old_assignments[range(batch_size), node_ids] = -1

        # Just return a new States with updated assignments
        return States(tensor=old_assignments)

    def reward(self, states: States) -> torch.Tensor:
        assignments = states.tensor
        batch_size = assignments.shape[0]
        rewards = torch.zeros(batch_size, dtype=torch.float, device=self.device)

        sink_mask = states.is_sink_state
        sink_assignments = assignments[sink_mask]

        if sink_assignments.numel() > 0:
            dist_pops = torch.zeros((sink_assignments.shape[0], self.num_districts),
                                    dtype=torch.float, device=self.device)
            for n in range(self.num_nodes):
                assigned_d = sink_assignments[:, n]
                valid_mask = (assigned_d != -1)
                if valid_mask.any():
                    dist_pops[valid_mask, assigned_d[valid_mask]] += self.populations[n]

            deviation = torch.mean(torch.abs(dist_pops - self.ideal_pop) / self.ideal_pop, dim=1)

            contiguity_penalties = torch.zeros(sink_assignments.shape[0], dtype=torch.float, device=self.device)
            for i in range(sink_assignments.shape[0]):
                state_assign = sink_assignments[i]
                if (state_assign == -1).any():
                    # If exit chosen before full assignment -> huge penalty
                    contiguity_penalties[i] = 999.0
                    continue
                penalty = 0
                for d_id in range(self.num_districts):
                    if not self._check_contiguity(state_assign.cpu().numpy(), d_id):
                        penalty += 1
                contiguity_penalties[i] = float(penalty)

            final_rewards = 1.0 / (1.0 + deviation + contiguity_penalties)
            rewards[sink_mask] = final_rewards

        return rewards

    def log_reward(self, states: States) -> torch.Tensor:
        r = self.reward(states)
        r = torch.clamp(r, min=1e-30)  # avoid log(0)
        return torch.log(r)

    def _check_contiguity(self, assignment: np.ndarray, district_id: int):
        district_nodes = np.where(assignment == district_id)[0]
        if len(district_nodes) <= 1:
            return True
        visited = set()
        to_visit = deque([district_nodes[0]])
        dist_set = set(district_nodes)
        while to_visit:
            cur = to_visit.popleft()
            visited.add(cur)
            for neigh in self.graph[cur]:
                if neigh in dist_set and neigh not in visited:
                    to_visit.append(neigh)
        return len(visited) == len(district_nodes)

    def preprocessor(self, states: torch.Tensor) -> torch.Tensor:
        # One-hot encode states: -1 -> class 0 (unassigned), districts 0..(num_districts-1) map to classes 1..num_districts
        state_onehot = torch.nn.functional.one_hot(states + 1, num_classes=self.num_districts + 1)
        return state_onehot.view(states.shape[0], -1).float()


if __name__ == "__main__":
    # Create environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = RedistrictEnv(num_districts=3, device=device)

    # Neural network input dimension
    input_dim = env.num_nodes * (env.num_districts + 1)  # one-hot encoding dimension

    # Forward (P_F) and Backward (P_B) policies
    module_PF = MLP(input_dim=input_dim, output_dim=env.n_actions).to(device)
    module_PB = MLP(input_dim=input_dim, output_dim=env.n_actions - 1, trunk=module_PF.trunk).to(device)

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # GFlowNet
    gfn = TBGFlowNet(logZ=0.0, pf=pf_estimator, pb=pb_estimator).to(device)

    # Sampler
    sampler = Sampler(estimator=pf_estimator)

    # Optimizer
    optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
    optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

    # Training Loop
    for i in (pbar := tqdm(range(1000))):
        trajectories = sampler.sample_trajectories(env=env, n=16)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})
