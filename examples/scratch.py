import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx


class GFlowNetPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # A simple feedforward network as a placeholder.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_repr):
        # state_repr: Tensor of shape [batch_size, state_dim]
        logits = self.net(state_repr)  # [batch_size, action_dim]
        # We'll convert logits to probabilities via a softmax:
        return torch.log_softmax(logits, dim=-1)


class RedistrictingEnvironment:
    def __init__(self, graph, num_districts):
        self.graph = graph
        self.num_districts = num_districts
        self.num_nodes = len(graph.nodes())
        # Additional attributes and preprocessing can be done here.

    def initial_state(self):
        # Return a representation of an empty assignment.
        # Example: state could be a vector of length num_nodes,
        # with -1 for unassigned and integers [0, num_districts-1] for assigned.
        # For a GFlowNet, we may want a more learned representation. For simplicity:
        state = torch.full((self.num_nodes,), -1, dtype=torch.long)
        return state

    def is_terminal(self, state):
        # Check if all units are assigned to a district
        return torch.all(state >= 0).item()

    def get_available_actions(self, state):
        # For simplicity:
        # Actions = { (node_to_assign, district_id) } for any unassigned node
        unassigned = (state == -1).nonzero().flatten().tolist()
        actions = []
        for node in unassigned:
            for d in range(self.num_districts):
                actions.append((node, d))
        return actions

    def apply_action(self, state, action):
        # action: (node, district)
        node, district = action
        new_state = state.clone()
        new_state[node] = district
        return new_state

    def state_to_feature(self, state):
        # Convert the state (assignments) into a feature vector suitable for the model.
        # This could be an embedding of node assignments, concatenated with graph features.
        # For simplicity, we’ll just one-hot encode the districts per node.
        # One-hot node assignments shape: [num_nodes * num_districts],
        # but to keep it simple, just return assignments as integers:
        # A more sophisticated representation would be needed in practice.
        one_hot = torch.zeros(self.num_nodes, self.num_districts)
        for i, d in enumerate(state):
            if d >= 0:
                one_hot[i, d] = 1.0
        return one_hot.flatten()

    def reward(self, final_state):
        # Define your custom reward. For example,
        # reward = some_function_of_compactness_and_fairness(final_state)
        # Return a scalar Tensor.
        # Here, just a dummy reward:
        return torch.tensor(1.0, requires_grad=False)


class GFlowNetTrainer:
    def __init__(self, env, policy, lr=1e-3):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def sample_trajectory(self):
        # Roll out a trajectory from the initial state to a terminal state.
        state = self.env.initial_state()
        trajectory = []

        while not self.env.is_terminal(state):
            # Get actions
            actions = self.env.get_available_actions(state)
            # Encode state and get action probabilities
            state_repr = self.env.state_to_feature(state).unsqueeze(0)
            log_probs = self.policy(state_repr)  # shape: [1, action_dim]

            # Map actions to an index in the policy output
            # For simplicity, let's assume a fixed ordering of actions
            # The action_dim is dynamic in practice, so you'd need a mask or dynamic indexing.
            # Instead, we dynamically compute them:
            # This is a simplification; a real GFlowNet would need a consistent mapping.

            # We'll build a mapping from action to index:
            actions_list = self.env.get_available_actions(state)
            action_dim = len(actions_list)
            # Re-compute log_probs for this step with correct action_dim
            # So we might need a separate linear layer or a method to handle dynamic action sets.
            # For demonstration, let's assume the policy outputs a fixed large action space
            # and we only index a subset. In a real scenario, you'd mask or dynamically
            # recompute probabilities for available actions.

            # For now, let's assume the policy always returns a vector large enough to cover
            # the maximum possible number of actions and we only use the first `action_dim`.
            curr_log_probs = log_probs[:, :action_dim]
            probs = curr_log_probs.exp()

            # Sample an action
            action_idx = torch.multinomial(probs.squeeze(0), num_samples=1).item()
            chosen_action = actions_list[action_idx]

            # Store the step
            trajectory.append((state, chosen_action, curr_log_probs[0, action_idx]))

            # Apply action
            state = self.env.apply_action(state, chosen_action)

        # Now we reached a terminal state
        final_reward = self.env.reward(state)
        return trajectory, final_reward

    def compute_trajectory_balance_loss(self, trajectory, final_reward):
        # Trajectory balance loss (simplified):
        #
        # If the trajectory is s_0 -> a_0 -> s_1 -> ... -> s_{n-1} -> a_{n-1} -> s_n,
        # with final reward R, and π_θ is the policy, then trajectory balance loss is:
        #
        # (log π_θ(a_0|s_0) + ... + log π_θ(a_{n-1}|s_{n-1})) - log R
        #
        # We want to minimize (this difference)^2, for example.

        log_prob_sum = sum([step[2] for step in trajectory])
        log_R = torch.log(final_reward)
        loss = (log_prob_sum - log_R).pow(2)
        return loss

    def train_step(self):
        self.optimizer.zero_grad()
        trajectory, final_reward = self.sample_trajectory()
        loss = self.compute_trajectory_balance_loss(trajectory, final_reward)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# Example usage:
if __name__ == "__main__":
    # Suppose we have a graph (pseudo-code, you would load a real graph)

    G = nx.grid_2d_graph(10, 10)  # 25 nodes arranged in a grid
    # Convert to a simple integer-labeled graph
    G = nx.convert_node_labels_to_integers(G)

    env = RedistrictingEnvironment(graph=G, num_districts=5)
    # state_dim: a simplistic assumption: each node * num_districts one-hot
    state_dim = len(G.nodes()) * 5
    # action_dim: max possible = (num_nodes * num_districts) in worst case
    # but actual available actions vary. For simplicity, define a large fixed action_dim:
    max_action_dim = env.num_nodes * env.num_districts

    policy = GFlowNetPolicy(state_dim=state_dim, action_dim=max_action_dim)
    trainer = GFlowNetTrainer(env, policy)

    for step in range(400):
        loss = trainer.train_step()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")