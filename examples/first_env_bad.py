import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


###########################################
# Environment Definition
###########################################

class RedistrictEnv:
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
            8: [5, 7]
        }
        self.populations = {0: 30, 1: 20, 2: 10, 3: 20, 4: 10, 5: 20, 6: 30, 7: 10, 8: 20}
        self.num_nodes = 9
        self.num_districts = num_districts
        self.nodes = list(self.graph.keys())

        total_pop = sum(self.populations.values())
        self.ideal_pop = total_pop / self.num_districts

        self.assignment = {}

    def initialize(self):
        pass

    def reset(self):
        self.assignment = {node: None for node in self.nodes}
        return self.get_state()

    def get_state(self):
        return np.array([self.assignment[n] if self.assignment[n] is not None else -1
                         for n in self.nodes], dtype=np.int64)

    def valid_actions(self, state):
        actions = []
        for node_id in range(self.num_nodes):
            if state[node_id] == -1:
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
        dist_pops = np.zeros(self.num_districts)
        for n, d in self.assignment.items():
            dist_pops[d] += self.populations[n]

        deviation = np.mean(np.abs(dist_pops - self.ideal_pop) / self.ideal_pop)

        contiguity_penalty = 0
        for d_id in range(self.num_districts):
            if not self._check_contiguity(d_id):
                contiguity_penalty += 1

        reward = 1.0 / (1.0 + deviation + contiguity_penalty)
        return reward

    def _check_contiguity(self, district_id):
        district_nodes = [n for n, d in self.assignment.items() if d == district_id]
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


###########################################
# GFlowNet Model
###########################################

class GFlowNetPolicy(nn.Module):
    def __init__(self, num_nodes, num_districts):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_districts = num_districts

        input_dim = num_nodes * (num_districts + 1)
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.action_head = nn.Linear(hidden_dim, num_nodes * num_districts)

    def forward(self, state):
        one_hot = []
        for val in state:
            vec = np.zeros(self.num_districts + 1, dtype=np.float32)
            if val == -1:
                vec[-1] = 1.0
            else:
                vec[val] = 1.0
            one_hot.append(vec)
        one_hot = np.concatenate(one_hot)
        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
        h = self.net(x)
        logits = self.action_head(h)
        return logits.squeeze(0)


class LogZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.logZ


###########################################
# GFlowNet Training (Trajectory Balance)
###########################################

def sample_trajectory(env, policy, logZ):
    state = env.reset()
    done = False

    states = []
    actions = []
    log_probs = []

    while not done:
        valid_acts = env.valid_actions(state)
        if len(valid_acts) == 0:
            break

        logits = policy(state)
        all_actions = [(n, d) for n in range(env.num_nodes) for d in range(env.num_districts)]
        mask = torch.zeros(len(all_actions), dtype=torch.bool)
        valid_set = set(valid_acts)
        idx_map = {}
        for i, a in enumerate(all_actions):
            if a in valid_set:
                mask[i] = True
                idx_map[len(idx_map)] = a

        valid_logits = logits[mask]
        action_dist = torch.distributions.Categorical(logits=valid_logits)
        action_idx = action_dist.sample()
        chosen_action = idx_map[action_idx.item()]

        chosen_global_idx = all_actions.index(chosen_action)
        chosen_logprob = logits[chosen_global_idx] - torch.logsumexp(logits[mask], dim=0)

        states.append(state.copy())
        actions.append(chosen_action)
        log_probs.append(chosen_logprob)

        state, reward, done, info = env.step(chosen_action)

    return states, actions, log_probs, reward


def evaluate_average_reward(env, policy, logZ, num_samples=100):
    rewards = []
    with torch.no_grad():
        for _ in range(num_samples):
            _, _, _, r = sample_trajectory(env, policy, logZ)
            rewards.append(r)
    return np.mean(rewards)


def train_gflownet(num_iterations=3000, print_every=300):
    env = RedistrictEnv(num_districts=3)
    policy = GFlowNetPolicy(num_nodes=env.num_nodes, num_districts=env.num_districts)
    logZ = LogZ()

    pre_train_reward = evaluate_average_reward(env, policy, logZ, num_samples=200)
    print("Average reward before training:", pre_train_reward)

    optimizer = optim.Adam(list(policy.parameters()) + list(logZ.parameters()), lr=0.01)

    for it in range(num_iterations):
        states, actions, log_probs, reward = sample_trajectory(env, policy, logZ)
        reward = max(reward, 1e-8)

        sum_log_pF = torch.stack(log_probs).sum()
        loss = (logZ() + sum_log_pF - np.log(reward)) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % print_every == 0:
            avg_r = evaluate_average_reward(env, policy, logZ, num_samples=200)
            print(f"Iter {it + 1}, Loss: {loss.item():.4f}, Avg reward: {avg_r:.4f}")

    post_train_reward = evaluate_average_reward(env, policy, logZ, num_samples=200)
    print("Average reward after training:", post_train_reward)

    return policy, logZ


if __name__ == "__main__":
    trained_policy, trained_logZ = train_gflownet()