from pathlib import Path
import numpy as np
import torch
from collections import deque
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_utils import load_raw_data


class RedistrictEnv:
    """
    A redistricting environment inspired by Imai et al. (2006).

    Each state is an assignment of precincts to districts. Initially, no precinct is assigned.
    Actions involve selecting an unassigned precinct and assigning it to a district.

    Once all precincts are assigned, a terminal reward is given based on:
        - Population deviation: how close each district is to the ideal population.
        - Contiguity: whether each district is contiguous.

    State representation:
        - A vector of length num_precincts, where each entry is the district assignment or -1 if unassigned.

    This environment is a starting point and can be improved by incorporating
    additional redistricting criteria or using actual data from the paper.
    """

    def __init__(
            self,
            graph: dict,
            populations: dict,
            num_districts: int = 3,
            device: Optional[torch.device] = None
    ):
        """
        Args:
            graph: A dictionary of {precinct_id: [adjacent_precinct_ids]}.
            populations: A dictionary {precinct_id: population}.
            num_districts: Number of districts to create.
            device: Torch device.
        """
        self.graph = graph
        self.populations = populations
        self.num_precincts = len(self.graph)
        self.precincts = list(self.graph.keys())
        self.num_districts = num_districts
        self.device = device if device is not None else torch.device("cpu")

        self.assignment = None
        self.n_actions = self.num_precincts * self.num_districts + 1  # assign any precinct to a district + exit
        total_pop = sum(self.populations.values())
        self.ideal_pop = total_pop / self.num_districts

    def reset(self):
        """
        Reset the environment to an initial state: all precincts unassigned.
        """
        self.assignment = {p: -1 for p in self.precincts}
        return self._get_state()

    def _get_state(self):
        """
        Return the current state as a numpy array of shape (num_precincts,).
        Each entry is the district of that precinct or -1 if unassigned.
        """
        return np.array([self.assignment[p] for p in self.precincts], dtype=np.int64)

    def valid_actions(self, state):
        """
        Returns a list of valid actions.
        Actions are tuples (precinct_id, district_id).
        Only unassigned precincts can be assigned to a district.

        We do not include the exit action here because it's only valid if all are assigned.
        """
        valid = []
        for i, precinct_id in enumerate(self.precincts):
            if state[i] == -1:
                for d_id in range(self.num_districts):
                    valid.append((precinct_id, d_id))
        # We could also add a terminating action if we want an early exit (but usually terminal is when all assigned)
        return valid

    def step(self, action):
        """
        Execute the action (precinct_id, district_id).
        Update the assignment and return (next_state, reward, done, info).
        """
        precinct_id, district_id = action
        self.assignment[precinct_id] = district_id
        state = self._get_state()
        done = self._is_terminal(state)
        reward = 0.0
        if done:
            reward = self._final_reward()
        return state, reward, done, {}

    def _is_terminal(self, state):
        # Terminal if all precincts have assigned districts
        return np.all(state != -1)

    def _final_reward(self):
        # Compute populations per district
        dist_pops = np.zeros(self.num_districts)
        for i, p in enumerate(self.precincts):
            d = self.assignment[p]
            dist_pops[d] += self.populations[p]

        # Compute population deviation
        # Imai et al. aim for equal population. The measure here:
        deviation = np.mean(np.abs(dist_pops - self.ideal_pop) / self.ideal_pop)

        # Compute contiguity penalty
        # We want each district to be a contiguous subgraph.
        # If a district is not contiguous, add a penalty.
        contiguity_penalty = 0
        for d_id in range(self.num_districts):
            if not self._check_contiguity(d_id):
                contiguity_penalty += 1

        # Combine into a final reward
        # Lower deviation and contiguity penalty => higher reward
        # This is a simple heuristic:
        reward = 1.0 / (1.0 + deviation + contiguity_penalty)
        return reward

    def _check_contiguity(self, district_id):
        # Extract precincts of this district
        district_nodes = [p for p, d in self.assignment.items() if d == district_id]
        if len(district_nodes) <= 1:
            return True  # single or no precinct is trivially contiguous

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

    def preprocessor(self, state):
        """
        Preprocess state for a policy network.
        We one-hot encode assignments: -1 (unassigned) -> 0 index, 1..num_districts for assigned.
        """
        state_tensor = torch.tensor(state, dtype=torch.long, device=self.device)
        one_hot = torch.nn.functional.one_hot(state_tensor + 1, num_classes=self.num_districts + 1)
        # Flatten: shape (num_precincts, num_districts+1) -> (num_precincts*(num_districts+1))
        return one_hot.view(-1).float()

    def sample_trajectories(self, n):
        """
        Sample n complete trajectories by randomly assigning precincts to districts until done.
        Returns a list of trajectories, where each trajectory is a list of (state, action, reward, next_state).
        """
        trajectories = []
        for _ in range(n):
            state = self.reset()
            trajectory = []
            done = False
            while not done:
                actions = self.valid_actions(state)
                action = actions[np.random.randint(len(actions))]
                next_state, reward, done, _ = self.step(action)
                trajectory.append((state, action, reward, next_state))
                state = next_state
            trajectories.append(trajectory)
        return trajectories


if __name__ == "__main__":

    # Example small graph: 4 precincts arranged in a line
    # 0--1--2--3
    graph = {
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
    populations = {0: 30, 1: 20, 2: 10, 3: 20, 4: 10, 5: 20, 6: 30, 7: 5, 8: 5}

    df = load_raw_data(Path('data/PA_raw_data.json'))

    # graph


    env = RedistrictEnv(graph, populations, num_districts=2)
    # Sample a few trajectories
    trajs = env.sample_trajectories(3)
    for t_i, traj in enumerate(trajs):
        print(f"Trajectory {t_i}:")
        for step in traj:
            s, a, r, s_next = step
            print(f"  State: {s}, Action: {a}, Reward: {r}")
