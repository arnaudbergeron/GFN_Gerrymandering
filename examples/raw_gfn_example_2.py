import numpy as np
from collections import defaultdict, deque


class RedistrictingEnv:
    def __init__(
            self,
            graph,  # dict: node_id -> list of neighbor_node_ids
            populations,  # dict: node_id -> population
            num_districts,  # int: number of districts
            ideal_pop=None,  # float: ideal population per district (e.g., total_pop/num_districts)
            pop_tolerance=0.05,  # float: allowable deviation fraction from ideal population
            contiguity_check=True,
            fairness_metric="population_deviation",
            max_steps=1000,  # for demonstration, to have a "done" condition if needed
    ):
        """
        Environment adapted to Imai, King, and Lau (2006).

        States are complete assignments of nodes to districts.
        Actions are local changes: pick a precinct and move it to an adjacent district.
        Each step checks contiguity and population constraints.

        Args:
            graph (dict): {node_id: [neighbors]}
            populations (dict): {node_id: population}
            num_districts (int): number of districts
            ideal_pop (float): desired (ideal) population per district. If None, computed from total.
            pop_tolerance (float): allowed relative deviation from ideal_pop.
            contiguity_check (bool): If True, ensure moves keep districts contiguous.
            fairness_metric (str): method for computing the reward/score.
                                   e.g., "population_deviation" or "population_variance".
            max_steps (int): maximum steps before done, optional.
        """
        self.graph = graph
        self.populations = populations
        self.nodes = list(graph.keys())
        self.num_nodes = len(self.nodes)
        self.num_districts = num_districts

        total_pop = sum(populations.values())
        if ideal_pop is None:
            self.ideal_pop = total_pop / self.num_districts
        else:
            self.ideal_pop = ideal_pop
        self.pop_tolerance = pop_tolerance

        self.contiguity_check = contiguity_check
        self.fairness_metric = fairness_metric
        self.max_steps = max_steps

        # State: {node_id: district_id}
        self.assignment = None
        self.step_count = 0

        # Precompute node indices
        self.node_index = {node: i for i, node in enumerate(self.nodes)}

        # Reset
        self.reset()

    def reset(self):
        """
        Reset the environment:
        Start from an initial valid districting plan.
        In practice, one might start from a known plan (e.g., a current official map)
        or generate a simple initial partition (like a breadth-first assignment).
        """
        # Simple initial partition: assign sequentially just to get a valid starting state.
        # NOTE: This might not be contiguous if done naively.
        # For demonstration, we try a BFS-based assignment from random seeds.
        self.assignment = self._generate_initial_valid_assignment()
        self.step_count = 0
        return self._get_observation()

    def _generate_initial_valid_assignment(self):
        # For simplicity, pick num_districts random seeds and grow districts with BFS.
        # This is a naive approach. In practice, one might load a known valid partition.
        np.random.seed(42)  # for reproducibility
        seeds = np.random.choice(self.nodes, self.num_districts, replace=False)
        assignment = {node: None for node in self.nodes}

        # BFS growth
        queues = []
        for d_id, seed in enumerate(seeds):
            assignment[seed] = d_id
            queues.append(deque([seed]))

        # Assign others by round-robin BFS until all assigned
        unassigned = set(self.nodes) - set(seeds)
        while unassigned:
            for d_id in range(self.num_districts):
                if not queues[d_id]:
                    # If no queue for this district, skip
                    continue
                current = queues[d_id].popleft()
                for neigh in self.graph[current]:
                    if neigh in unassigned:
                        assignment[neigh] = d_id
                        unassigned.remove(neigh)
                        queues[d_id].append(neigh)
                        if not unassigned:
                            break
                if not unassigned:
                    break

        return assignment

    def _get_observation(self):
        # Return current assignment as the state observation.
        return dict(self.assignment)

    def step(self, action):
        """
        Perform an action:
        Action format: (node_id, new_district_id)

        This attempts to move node_id from its current district to new_district_id,
        provided it maintains validity (contiguity, population constraints).
        """
        node_id, new_district_id = action

        old_district_id = self.assignment[node_id]
        if old_district_id == new_district_id:
            # No change
            return self._get_observation(), 0.0, self.is_terminal(), {}

        # Check validity of this action
        if not self._is_valid_action(node_id, new_district_id):
            # Invalid action, you might raise an error or return a penalty.
            # We'll just raise an error for now.
            raise ValueError(
                "Invalid action: cannot move node {} to district {}".format(
                    node_id, new_district_id
                )
            )

        # Apply the action
        self.assignment[node_id] = new_district_id

        # In MCMC, typically you don't have a "done" state unless you run a fixed number of iterations.
        # For GFlowNets, you might define done as having a final assignment. Here we define done after max_steps.
        self.step_count += 1
        done = self.is_terminal()

        # Compute reward:
        # For GFlowNets, this might be final (only if done).
        # For MCMC, you might want to always get a "score" that corresponds to the stationary distribution.
        reward = self.final_reward() if done else 0.0

        return self._get_observation(), reward, done, {}

    def is_terminal(self):
        # Define a terminal condition for demonstration:
        # For GFlowNets: If we consider one full plan as a terminal state, we can say done=True immediately.
        # For MCMC: Typically no terminal state, but we can use max_steps.
        return self.step_count >= self.max_steps

    def final_reward(self):
        """
        Compute a final reward for the current partition.

        The paper discusses generating representative samples of plans.
        A neutral approach: reward = 1 / (1 + population_deviation)
        to encourage more balanced plans.
        """
        dist_pops = self._district_populations()

        if self.fairness_metric == "population_deviation":
            # deviation = sum(|pop - ideal_pop|) / (num_districts * ideal_pop)
            deviation = np.mean(np.abs(dist_pops - self.ideal_pop) / self.ideal_pop)
            return 1.0 / (1.0 + deviation)
        elif self.fairness_metric == "population_variance":
            var = np.var(dist_pops)
            return 1.0 / (1.0 + var)
        else:
            # Default: uniform reward
            return 1.0

    def _is_valid_action(self, node_id, new_district_id):
        """
        Check whether reassigning node_id to new_district_id yields a valid plan.

        Validity includes:
        - Population balance within tolerance
        - Contiguity of the affected districts
        """
        old_district_id = self.assignment[node_id]

        # Check population constraints first
        # We compute new populations if we do the reassignment
        dist_pops = self._district_populations()
        dist_pops[old_district_id] -= self.populations[node_id]
        dist_pops[new_district_id] += self.populations[node_id]

        if not self._check_population_balance(dist_pops):
            return False

        if self.contiguity_check:
            # Temporarily assign and check contiguity
            original_assignment = self.assignment[node_id]
            self.assignment[node_id] = new_district_id

            # Check contiguity for the old district (after removal of node_id)
            if not self._check_contiguity(old_district_id):
                self.assignment[node_id] = original_assignment
                return False

            # Check contiguity for the new district (after addition of node_id)
            if not self._check_contiguity(new_district_id):
                self.assignment[node_id] = original_assignment
                return False

            # Restore assignment
            self.assignment[node_id] = original_assignment

        return True

    def _district_populations(self):
        """
        Compute the population of each district from the current assignment.
        """
        dist_pops = np.zeros(self.num_districts)
        for node_id, d_id in self.assignment.items():
            dist_pops[d_id] += self.populations[node_id]
        return dist_pops

    def _check_population_balance(self, dist_pops):
        """
        Ensure each district's population is within the tolerance of the ideal population.
        """
        lower_bound = self.ideal_pop * (1 - self.pop_tolerance)
        upper_bound = self.ideal_pop * (1 + self.pop_tolerance)
        return np.all((dist_pops >= lower_bound) & (dist_pops <= upper_bound))

    def _check_contiguity(self, district_id):
        """
        Check that the subgraph induced by all nodes in district_id is contiguous.
        We do a BFS from one node in the district and ensure we can reach all others.
        """
        # Extract nodes in this district
        district_nodes = [n for n, d in self.assignment.items() if d == district_id]
        if len(district_nodes) <= 1:
            return True  # trivially contiguous if 0 or 1 node

        visited = set()
        to_visit = deque([district_nodes[0]])
        district_set = set(district_nodes)

        while to_visit:
            current = to_visit.popleft()
            visited.add(current)
            for neigh in self.graph[current]:
                if neigh in district_set and neigh not in visited:
                    to_visit.append(neigh)

        return len(visited) == len(district_nodes)

    def valid_actions(self):
        """
        List all valid actions from the current state.
        An action is (node_id, new_district_id).
        Here we consider actions that move a node to an adjacent district
        (i.e., a district of one of its neighbors).
        """
        actions = []
        for node_id, d_id in self.assignment.items():
            # Determine candidate districts by looking at neighbors
            candidate_districts = set()
            for neigh in self.graph[node_id]:
                neigh_d_id = self.assignment[neigh]
                if neigh_d_id != d_id:
                    candidate_districts.add(neigh_d_id)

            for new_d_id in candidate_districts:
                if self._is_valid_action(node_id, new_d_id):
                    actions.append((node_id, new_d_id))
        return actions


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


###########################################
# Environment Definition
###########################################

class SimpleRedistrictEnv:
    """
    A simple environment for demonstration:
    - 5 nodes in a line: 0--1--2--3--4
    - 2 districts.
    - State: assignment of each node to {0,1} or None.
    - Actions: assign an unassigned node to a district.
    - Reward: higher if populations are balanced.
    - No contiguity checks here for simplicity.
    """

    def __init__(self, num_districts=2):
        self.graph = {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3]
        }
        self.populations = {i: (10 + i) for i in range(5)}  # slightly varying populations
        self.nodes = list(self.graph.keys())
        self.num_nodes = len(self.nodes)
        self.num_districts = num_districts
        self.assignment = None

        # Precompute ideal population for reward
        total_pop = sum(self.populations.values())
        self.ideal_pop = total_pop / self.num_districts

    def reset(self):
        self.assignment = {node: None for node in self.nodes}
        return self.get_state()

    def get_state(self):
        # State as a vector: for each node, district_id or -1 if None
        # shape: [num_nodes], each value in {-1,0,...,num_districts-1}
        arr = np.array([self.assignment[n] if self.assignment[n] is not None else -1
                        for n in self.nodes], dtype=np.int64)
        return arr

    def valid_actions(self, state):
        # Actions: (node_id, district_id)
        # Node must be unassigned
        actions = []
        for node_id in self.nodes:
            if state[node_id] == -1:  # unassigned
                for d_id in range(self.num_districts):
                    actions.append((node_id, d_id))
        return actions

    def step(self, action):
        # action: (node_id, district_id)
        node_id, district_id = action
        self.assignment[node_id] = district_id
        state = self.get_state()
        done = self.is_terminal(state)
        reward = 0.0
        if done:
            reward = self.final_reward()
        return state, reward, done, {}

    def is_terminal(self, state):
        # Terminal if all assigned
        return np.all(state != -1)

    def final_reward(self):
        # Compute population in each district
        dist_pops = np.zeros(self.num_districts)
        for n, d_id in self.assignment.items():
            dist_pops[d_id] += self.populations[n]
        # Reward: 1/(1+variance) or 1/(1+average deviation)
        deviation = np.mean(np.abs(dist_pops - self.ideal_pop) / self.ideal_pop)
        reward = 1 / (1 + deviation)
        return reward


###########################################
# GFlowNet Model
###########################################

class GFlowNetPolicy(nn.Module):
    """
    A simple GFlowNet policy network.
    Input: state (assignments)
    Output: a distribution over actions.
    We will represent the state as a one-hot encoding of districts for each node:
      - unassigned = a special vector
      - assigned = one-hot of dimension num_districts

    For simplicity, just a small MLP.
    """

    def __init__(self, num_nodes, num_districts):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_districts = num_districts

        input_dim = num_nodes * (num_districts + 1)  # each node: one-hot for dists + "unassigned"
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # We'll produce logits for all possible actions: num_nodes * num_districts
        # But we can mask invalid actions at sampling/training time.
        self.action_head = nn.Linear(hidden_dim, num_nodes * num_districts)

    def forward(self, state):
        # state: shape [num_nodes], with values in {-1,d1,...,dK-1}
        # We'll convert this to a one-hot:
        # If assigned to district d, one-hot vector of length (num_districts+1) with a 1 in position d
        # If unassigned (-1), one-hot vector with a 1 in the last position (num_districts)
        one_hot = []
        for val in state:
            vec = np.zeros(self.num_districts + 1, dtype=np.float32)
            if val == -1:
                vec[-1] = 1.0
            else:
                vec[val] = 1.0
            one_hot.append(vec)
        one_hot = np.concatenate(one_hot)  # shape: num_nodes*(num_districts+1)

        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)  # [1, input_dim]
        h = self.net(x)
        logits = self.action_head(h)  # [1, num_nodes*num_districts]
        return logits.squeeze(0)  # [num_nodes*num_districts]


###########################################
# GFlowNet Training Setup (Trajectory Balance)
###########################################

# We will implement the trajectory balance objective:
# loss = (log(Z) + sum_t log P_F(a_t|s_t) - log R)²
# where P_F is the forward policy (our model), and Z is a learned scalar.

class LogZ(nn.Module):
    # A learnable scalar for log Z
    def __init__(self):
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.logZ


def sample_trajectory(env, policy, logZ):
    """
    Sample a trajectory from the GFlowNet:
    Start from s0 = empty assignment
    At each state, sample actions from the policy.
    """
    state = env.reset()
    done = False

    states = []
    actions = []
    log_probs = []

    while not done:
        # Get valid actions
        valid_acts = env.valid_actions(state)
        if len(valid_acts) == 0:
            # No valid action (shouldn't happen if we can always assign)
            # but if it does, terminate
            break

        # Compute logits
        logits = policy(state)
        # Mask invalid actions
        # valid_actions_index: a mapping from the action (node_id,d_id) to index
        all_actions = [(n, d) for n in range(env.num_nodes) for d in range(env.num_districts)]
        mask = torch.zeros(len(all_actions), dtype=torch.bool)
        valid_set = set(valid_acts)
        for i, a in enumerate(all_actions):
            if a in valid_set:
                mask[i] = True

        valid_logits = logits[mask]  # only valid actions
        action_dist = torch.distributions.Categorical(logits=valid_logits)
        action_idx = action_dist.sample()  # index in valid_actions

        chosen_action = list(valid_set)[action_idx.item()]
        # map chosen_action back to global index if needed
        # Actually, we have chosen_action directly.

        # Compute log_prob of this chosen action
        # We must find the index in all_actions that corresponds to chosen_action
        chosen_global_idx = all_actions.index(chosen_action)
        chosen_logits = logits[chosen_global_idx]
        chosen_logprob = chosen_logits - torch.logsumexp(logits[mask], dim=0)

        states.append(state.copy())
        actions.append(chosen_action)
        log_probs.append(chosen_logprob)

        # Step
        state, reward, done, info = env.step(chosen_action)

    return states, actions, log_probs, reward


def train_gflownet(num_iterations=2000, print_every=200):
    env = SimpleRedistrictEnv(num_districts=2)
    policy = GFlowNetPolicy(num_nodes=5, num_districts=2)
    logZ = LogZ()

    optimizer = optim.Adam(list(policy.parameters()) + list(logZ.parameters()), lr=0.01)

    for it in range(num_iterations):
        states, actions, log_probs, reward = sample_trajectory(env, policy, logZ)
        reward = max(reward, 1e-8)  # avoid log(0)

        # TB loss
        sum_log_pF = torch.stack(log_probs).sum()
        loss = (logZ() + sum_log_pF - np.log(reward)) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % print_every == 0:
            print(
                f"Iter {it + 1}, Loss: {loss.item():.4f}, Reward: {reward:.4f}, exp(logZ): {torch.exp(logZ()).item():.4f}")

    return policy, logZ


# Example usage:
if __name__ == "__main__":
    # Small example graph: A line of 5 nodes
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
    populations = {i: 10 for i in range(5)}
    num_districts = 2
    env = RedistrictingEnv(graph, populations, num_districts, pop_tolerance=0.5)

    state = env.reset()
    print("Initial state:", state)
    print("Initial valid actions:", env.valid_actions())

    done = False
    while not done:
        valid_actions = env.valid_actions()
        if not valid_actions:
            print("No valid actions left.")
            break
        action = valid_actions[0]
        state, reward, done, info = env.step(action)
        print("Took action:", action, "Reward:", reward, "Done:", done)

    # Training gflownets
    trained_policy, trained_logZ = train_gflownet()

    # After training, we can sample solutions and see if they tend to have high rewards.
    env = SimpleRedistrictEnv(num_districts=2)
    rewards = []
    for _ in range(100):
        _, _, _, r = sample_trajectory(env, trained_policy, trained_logZ)
        rewards.append(r)
    print("Average reward after training:", np.mean(rewards))
