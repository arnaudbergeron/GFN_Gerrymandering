# https://github.com/GFNOrg/torchgfn/tree/master
from collections import deque

from networkx import number_of_nodes
import torch
from tqdm import tqdm
import numpy as np
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP  # is a simple multi-layer perceptron (MLP)


class RedistrictEnv:
    """
    Precincts: 0 through 8 arranged as:
        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
        |    |    |
        6 -- 7 -- 8

    Populations = [30,20,10,20,10,20,30,10], sum=150
    num_districts=3, ideal_pop=50 each

    Goal: Assign nodes to 3 districts so that each district is contiguous and has population ~30.

    Reward = 1/(1+pop_deviation+contiguity_penalty)
    pop_deviation = mean(|dist_pop - 30|/30)
    contiguity_penalty = number_of_noncontiguous_districts
    """

    def __init__(self, num_districts=3):
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
        self.populations = {0: 30, 1: 20, 2: 10, 3: 20, 4: 10, 5: 20, 6: 30, 7: 10}
        self.num_nodes = 8
        self.num_districts = num_districts
        self.nodes = list(self.graph.keys())

        """
        States:
        - forward_masks: actions allowed (batch_shape, n_actions)
        - backward_masks: actions that could have led to each state (batch_shape, n_actions-1)
        
        
        Actions: indexed from 0 to n_actions - 1
        (*batch_shape, *action_shape)
        - last action is exit or terminate action: s -> s_f
        - dummy action : [-1]
        - exit_action: [n_actions - 1]
        
        Containers:
        - collection of states along with reward values, densities
        - can be instantiated with states object or nothing, to populate on the fly for replay buffer
        - to sample complete trajectories, can use .to_transitions() and .to_states()
        
        Types of containers:
        - transitions: batch s-> s'
        - trajectories: tau = s_0 -> s_1 -> ...
        
        
        Modules:
        - DiscretePolicyEstimator - P_F and P_B
        - when is_backward=False, use n_actions for dim
        - when is_backward=True, use n_actions for dim
        
        
        """
        self.n_actions = self.num_nodes * self.num_nodes + 1  # num_nodes + exit action

        total_pop = sum(self.populations.values())
        self.ideal_pop = total_pop / self.num_districts

        self.assignment = {}

    def reset(self):
        self.assignment = {node: None for node in self.nodes}
        return self.get_state()

    def preprocessor(self, state):
        """Raw state to tensors"""
        return torch.tensor(state, dtype=torch.float32)

    def get_state(self):
        # State as an array of length num_nodes: district_id or -1 if None
        return np.array(
            [
                self.assignment[n] if self.assignment[n] is not None else -1
                for n in self.nodes
            ],
            dtype=np.int64,
        )

    def valid_actions(self, state):
        actions = []
        for node_id in range(self.num_nodes):
            if state[node_id] == -1:  # unassigned
                for d_id in range(self.num_districts):
                    actions.append((node_id, d_id))
        return actions

    def step(self, action):
        node_id, district_id = action
        self.assignment[node_id] = district_id
        state = self.get_state()
        done = self.is_terminal(state)
        reward = 0.0
        if done:
            reward = self.final_reward()
        return state, reward, done, {}

    def is_terminal(self, state):
        return np.all(state != -1)

    def final_reward(self):
        # Compute district populations
        dist_pops = np.zeros(self.num_districts)
        for n, d in self.assignment.items():
            dist_pops[d] += self.populations[n]

        # population deviation
        deviation = np.mean(np.abs(dist_pops - self.ideal_pop) / self.ideal_pop)

        # contiguity check
        contiguity_penalty = 0
        for d_id in range(self.num_districts):
            if not self._check_contiguity(d_id):
                contiguity_penalty += 1

        reward = 1.0 / (1.0 + deviation + contiguity_penalty)
        return reward

    def _check_contiguity(self, district_id):
        # Extract nodes of this district
        district_nodes = [n for n, d in self.assignment.items() if d == district_id]
        if len(district_nodes) <= 1:
            return True  # single node or empty is trivially contiguous
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

    def sample_trajectories(self, n):
        trajectories = []
        for _ in range(n):
            state = self.reset()
            trajectory = []
            done = False
            while not done:
                actions = self.valid_actions(state)
                action = actions[np.random.choice(len(actions))]
                next_state, reward, done, _ = self.step(action)
                trajectory.append((state, action, reward, next_state))
                state = next_state
            trajectories.append(trajectory)
        return trajectories


def transform_states_to_tensors():
    """
    - Keep track also of batch_shape = (n_states, n_traj)
    - append dummy to traj shorter than longest traj
    """
    pass


# 1 - We define the environment.
env = RedistrictEnv(num_districts=3)

# 2 - We define the needed modules (neural networks).
# The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
input_dim = env.num_nodes * 2
module_PF = MLP(
    input_dim=input_dim,
    output_dim=env.n_actions,
)  # Neural network for the forward policy, with as many outputs as there are actions

module_PB = MLP(
    input_dim=input_dim,
    output_dim=env.n_actions - 1,
    trunk=module_PF.trunk,  # We share all the parameters of P_F and P_B, except for the last layer
)

# 3 - We define the estimators.
pf_estimator = DiscretePolicyEstimator(
    module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor
)
pb_estimator = DiscretePolicyEstimator(
    module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor
)

# 4 - We define the GFlowNet.
gfn = TBGFlowNet(logZ=0.0, pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

# 5 - We define the sampler and the optimizer.
sampler = Sampler(
    estimator=pf_estimator
)  # We use an on-policy sampler, based on the forward policy

# Different policy parameters can have their own LR.
# Log Z gets dedicated learning rate (typically higher).
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

# 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
for i in (pbar := tqdm(range(1000))):
    trajectories = sampler.sample_trajectories(env=env, n=16)
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        pbar.set_postfix({"loss": loss.item()})
