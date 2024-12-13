import random
from scipy.stats import poisson
import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
from shapely.geometry import Polygon, Point

from utils.data_utils import *

# Create a persistent district-to-color mapping
DISTRICT_COLOR_MAP = {}


def initialize_district_color_map(unique_districts, cmap_name="tab20"):
    """
    Initialize the district-to-color mapping with a consistent colormap.
    """
    global DISTRICT_COLOR_MAP
    base_cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cmap = ListedColormap(base_cmap.colors[:len(unique_districts)])
    DISTRICT_COLOR_MAP = {
        district: to_hex(cmap(i))
        for i, district in enumerate(sorted(unique_districts))
    }


def visualize_map_with_graph_and_geometry(G, E_on, boundary_components, df, district_id_col,
                                          geometry_col="geometry"):
    """
    Combines the map visualization with the graph overlay, highlighting boundary-connected nodes with inverse colors.
    """
    global DISTRICT_COLOR_MAP

    # Ensure the district color map is initialized
    unique_districts = df[district_id_col].unique()
    if not DISTRICT_COLOR_MAP:
        initialize_district_color_map(unique_districts)

    # Set up figure with reduced margins
    fig, ax = plt.subplots(figsize=(15, 12))

    # Plot the map with districts
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    gdf['color'] = gdf[district_id_col].map(DISTRICT_COLOR_MAP)
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black')

    # Create a set for quick lookup of E_on edges
    E_on_set = {(min(u, v), max(u, v)) for u, v in E_on}

    # Ensure all nodes have coordinates
    coordinates = df.set_index('node_id')['coordinates'].to_dict()
    pos = {node: coordinates.get(node, (0, 0)) for node in G.nodes()}

    # Partition
    partition = df.set_index('node_id')[district_id_col].to_dict()

    # Map nodes to their district colors and apply inverse color for boundary nodes
    node_color_map = []
    for node in G.nodes():
        if node in partition:  # Ensure the node is part of the partition
            district_color = DISTRICT_COLOR_MAP.get(partition[node], "#CCCCCC")
            if any(node in comp for comp in boundary_components):
                # Calculate inverse color for boundary-connected nodes
                r, g, b, _ = mcolors.to_rgba(district_color)
                inverse_color = to_hex((1 - r, 1 - g, 1 - b))
                node_color_map.append(inverse_color)
            else:
                node_color_map.append(district_color)
        else:
            node_color_map.append("#CCCCCC")  # Default color for nodes not in the partition

    # Edge colors: white for default edges, red for E_on edges
    edge_color_map = ['red' if (min(u, v), max(u, v)) in E_on_set else 'white' for u, v in G.edges()]
    edge_widths = [3 if (min(u, v), max(u, v)) in E_on_set else 2 for u, v in G.edges()]

    # Overlay the graph on the map with larger nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_map, node_size=500, edgecolors='black', alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color_map, width=edge_widths, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')

    # Add a legend in the top-right corner
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='E_on edges'),
        Line2D([0], [0], color='white', lw=2, label='Other edges'),
        Patch(facecolor="#CCCCCC", edgecolor='black', label='Non-boundary nodes'),
    ]
    for district, color in DISTRICT_COLOR_MAP.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'District {district}'))

    state = df['state'][0]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Combined Map and Graph Visualization for {state} State", fontsize=16)

    # Adjust layout to minimize white space
    plt.tight_layout(pad=1.0)
    plt.show()


def population_equality_reward(partition, populations):
    # Compute district populations
    districts = set(partition.values())
    district_pops = {d: 0 for d in districts}
    for node, dist in partition.items():
        district_pops[dist] += populations[node]

    # Compute the ideal population
    total_pop = sum(populations.values())
    ideal_pop = total_pop / len(districts)

    # Compute sum of squared deviations from ideal
    ssd_pop = 0.0
    for dist in districts:
        diff = district_pops[dist] - ideal_pop
        ssd_pop += diff * diff

    # Return negative of SSD for population equality
    return -ssd_pop


def voting_share_reward(partition, votes_dem, votes_rep):
    # Compute district-level vote shares
    districts = set(partition.values())
    district_dem_votes = {d: 0 for d in districts}
    district_total_votes = {d: 0 for d in districts}

    for node, dist in partition.items():
        district_dem_votes[dist] += votes_dem[node]
        district_total_votes[dist] += (votes_dem[node] + votes_rep[node])

    # Compute deviation from 50% dem share
    ssd_vote = 0.0
    for dist in districts:
        if district_total_votes[dist] > 0:
            dem_share = district_dem_votes[dist] / district_total_votes[dist]
            diff = dem_share - 0.5
            ssd_vote += diff * diff
        else:
            # If a district has no votes at all, treat as perfectly balanced
            # or handle differently if needed
            pass

    # Return negative of SSD for voting share balance
    return -ssd_vote


def combined_reward(partition, populations, votes_dem, votes_rep, w_pop=1.0, w_vote=1.0):
    # Combine both population equality and voting share balance
    pop_score = population_equality_reward(partition, populations)
    vote_score = voting_share_reward(partition, votes_dem, votes_rep)
    return w_pop * pop_score + w_vote * vote_score


def run_algorithm_1(df, q, num_iterations, lambda_param=2):
    """
    Executes the redistricting algorithm for the specified number of iterations.

    Parameters:
    - df: DataFrame containing the nodes and their attributes.
    - q: Probability threshold for turning on edges.
    - num_iterations: Number of iterations to run the algorithm.
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (used in select_nonadjacent_components).

    Returns:
    - samples: List of partition samples after each iteration.
    """
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Precompute node-level data for reward calculations
    populations = df.set_index('node_id')['pop'].to_dict()
    votes_dem = df.set_index('node_id')['pre_20_dem_bid'].to_dict()
    votes_rep = df.set_index('node_id')['pre_20_rep_tru'].to_dict()
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

    # Initialize the partition and store initial results for visualization
    current_partition = initial_partition.copy()
    district = np.array([current_partition[node] for node in range(len(df))])
    df['district'] = district

    def g_func(partition):
        """
        Compute the unnormalized target distribution g(π).
        Combine population equality and voting share rewards.
        """
        pop_score = population_equality_reward(partition, populations)
        vote_score = voting_share_reward(partition, votes_dem, votes_rep)
        return np.exp(pop_score + vote_score)  # Exponentiate to simulate a probability distribution

    # List to store partition samples
    samples = []

    # Iterative algorithm
    for t in range(num_iterations):
        # Step 1: Determine E_on edges
        E_on = turn_on_edges(G, current_partition, q)

        # Step 2: Find boundary components
        boundary_components = find_boundary_connected_components(G, current_partition, E_on)

        # Step 3: Select nonadjacent components using the updated method
        V_CP = select_nonadjacent_components(boundary_components, G, current_partition, lambda_param=lambda_param)

        # Step 4: Propose swaps for the selected components
        proposed_partition = propose_swaps(current_partition, V_CP, G)

        # Update the DataFrame with the new partition for visualization
        district = np.array([proposed_partition[node] for node in range(len(df))])
        df['district'] = district

        # Step 5: Accept or reject the proposed partition
        current_partition = accept_or_reject_proposal(
            current_partition, proposed_partition, V_CP, q, g_func
        )

        # # Step 5: Evaluate rewards
        # current_reward = combined_reward(current_partition, populations, votes_dem, votes_rep)
        # proposed_reward = combined_reward(proposed_partition, populations, votes_dem, votes_rep)
        #
        # # Step 6: Accept or reject the proposed partition
        # if proposed_reward >= current_reward:
        #     current_partition = proposed_partition

        # Step 7: Save the current partition
        samples.append(current_partition.copy())

        # Visualization (optional; can slow down execution if num_iterations is large)
        visualize_map_with_graph_and_geometry(G, E_on, boundary_components, df, 'district')

    return samples


def turn_on_edges(G, partition, q):
    """
    Turns on edges that are in the same district with probability q.
    Used to make random connected components in the graph, or small sub-graphs inside a district.
    """
    E_on = []
    for u, v in G.edges():
        if partition[u] == partition[v] and random.random() < q:
            E_on.append((u, v))
    return E_on


def find_boundary_connected_components(G, partition, E_on):
    """
    Identify boundary components based on E_on edges.
    Components are valid if at least one node in the component has a neighbor
    in a different district.
    """
    # Step 1: Build a subgraph of E_on edges
    boundary_subgraph = G.edge_subgraph(E_on).copy()

    # Step 2: Get connected components from E_on subgraph
    connected_components = list(nx.connected_components(boundary_subgraph))

    # Step 3: Filter components that are on the boundary
    filtered_components = []
    for component in connected_components:
        if is_component_on_boundary(component, partition, G):
            filtered_components.append(component)

    return filtered_components


def identify_boundary_nodes(graph, partition):
    boundary_nodes = set()
    for node in graph:
        current_district = partition[node]
        for neigh in graph[node]:
            if partition[neigh] != current_district:
                boundary_nodes.add(node)
                break
    return boundary_nodes


def is_component_on_boundary(component, partition, G):
    """
    Check if a component has any node with a neighbor in a different district.
    """
    component_district = {partition[node] for node in component}
    if len(component_district) != 1:
        return False  # The component spans multiple districts, which is invalid

    component_district = component_district.pop()
    for node in component:
        for neigh in G.neighbors(node):
            if neigh not in component and partition[neigh] != component_district:
                return True  # A node in the component is adjacent to a different district
    return False


def precompute_district_nodes(partition, G):
    """
    Precompute a mapping of districts to their nodes.

    Parameters:
    - partition: Current district assignments.
    - G: The graph (NetworkX object).

    Returns:
    - Dictionary mapping districts to their nodes.
    """
    district_nodes = {}
    for node in G.nodes():
        district = partition[node]
        if district not in district_nodes:
            district_nodes[district] = set()
        district_nodes[district].add(node)
    return district_nodes


def select_nonadjacent_components(boundary_components, G, partition, lambda_param=2):
    """
    Select nonadjacent connected components along boundaries using the logic described in Step 3.

    Parameters:
    - boundary_components: List of connected components along the boundary.
    - G: The graph (NetworkX object).
    - partition: Current district assignments.
    - lambda_param: Lambda for the zero-truncated Poisson distribution.

    Returns:
    - V_CP: Selected nonadjacent connected components.
    """
    if len(boundary_components) == 0:
        return []

    # Precompute district nodes
    district_nodes = precompute_district_nodes(partition, G)

    # Step 3(a): Generate R from a zero-truncated Poisson distribution
    max_components = len(boundary_components)
    R = 0
    while R < 1 or R > max_components:
        R = poisson.rvs(lambda_param)  # Sample from Poisson
    R = min(R, max_components)  # Ensure R does not exceed |B(CP, π)|

    # Step 3(b): Initialize V_CP and start selecting components
    V_CP = set()
    V_CP_list = []
    remaining_components = {frozenset(component) for component in boundary_components}

    while len(V_CP) < R and remaining_components:
        C = set(random.sample(remaining_components, 1)[0])  # Randomly sample without replacement
        remaining_components.remove(frozenset(C))

        # Check if C is adjacent to any component in V_CP or causes noncontiguous districts
        if not is_adjacent_to_vcp(C, V_CP, G) and not causes_noncontiguous_district(V_CP | C, district_nodes, G):
            V_CP |= C  # Add C to V_CP only if it satisfies all constraints
            V_CP_list.append(tuple(C))

    return V_CP_list


def is_adjacent_to_vcp(C, V_CP, G):
    """
    Check if component C is adjacent to any component in V_CP.

    Parameters:
    - C: Component to check.
    - V_CP: Set of selected components.
    - G: The graph (NetworkX object).

    Returns:
    - Boolean indicating if C is adjacent to any component in V_CP.
    """
    if not V_CP:
        return False
    for node in C:
        for neighbor in G.neighbors(node):
            for other_component in V_CP:
                if neighbor == other_component:
                    return True
    return False


def causes_noncontiguous_district(V_CP, district_nodes, G):
    """
    Check if the removal of V_CP causes a noncontiguous district.

    Parameters:
    - V_CP: Set of selected components to be removed.
    - district_nodes: Precomputed mapping of districts to their nodes.
    - G: The graph (NetworkX object).

    Returns:
    - Boolean indicating if removing V_CP causes a noncontiguous district.
    """
    if not V_CP:
        return False  # If no components are selected, no districts can become noncontiguous

    for district, nodes in district_nodes.items():
        remaining_nodes = nodes - V_CP
        if remaining_nodes:
            subgraph = G.subgraph(remaining_nodes)
            if not nx.is_connected(subgraph):
                return True  # Noncontiguous district found
    return False


def propose_swaps(current_partition, V_CP, G):
    """
    Propose swaps for the selected components (V_CP) following the paper's logic.

    Parameters:
    - current_partition: The current partitioning of nodes into districts.
    - V_CP: Selected connected components to be reassigned.
    - G: The graph (NetworkX object).

    Returns:
    - proposed_partition: The updated partition after proposing swaps.
    """
    proposed_partition = current_partition.copy()

    for component in V_CP:
        # (a) Determine the current district of the component
        current_district_set = {proposed_partition[node] for node in component}
        if len(current_district_set) != 1:
            continue  # Skip if the component spans multiple districts
        current_district = current_district_set.pop()

        # Find neighboring districts for the component
        neighbor_districts = find_neighboring_districts(component, proposed_partition, G)
        if not neighbor_districts:
            continue  # Skip if no neighboring districts are found

        # Randomly select a new neighboring district
        new_district = random.choice(neighbor_districts)

        # (b) Update the partition: Remove C from current district and add to new district
        for node in component:
            proposed_partition[node] = new_district

    return proposed_partition


def find_neighboring_districts(component, partition, G):
    """
    Find neighboring districts for a given component.

    Parameters:
    - component: The set of nodes in the component.
    - partition: The current partitioning of nodes into districts.
    - G: The graph (NetworkX object).

    Returns:
    - A list of neighboring districts.
    """
    current_district = {partition[n] for n in component}.pop()  # Current district of the component
    adjacent_districts = set()

    for node in component:
        for neighbor in G.neighbors(node):
            if neighbor not in component and partition[neighbor] != current_district:
                adjacent_districts.add(partition[neighbor])

    return list(adjacent_districts)


def compute_acceptance_probability(current_partition, proposed_partition, V_CP, q, g_func):
    """
    Computes the acceptance probability α(π', CP | π).

    Parameters:
    - current_partition: Current partition π (dict mapping node to district).
    - proposed_partition: Proposed partition π'.
    - V_CP: Set of connected components selected in Step 3.
    - q: Probability threshold for turning on edges.
    - g_func: Function to compute g(π), the unnormalized target distribution for sampling.

    Returns:
    - alpha: Acceptance probability for the proposal.
    """
    # Probability of forming V_CP in current and proposed partitions
    q_forward = (q ** len(V_CP))
    q_backward = (q ** len(V_CP))  # Symmetric proposal

    # Compute g(π') / g(π)
    g_current = g_func(current_partition)
    g_proposed = g_func(proposed_partition)

    if g_current == 0:
        return 1.0 if g_proposed > 0 else 0.0  # Handle edge cases

    g_ratio = g_proposed / g_current

    # Compute alpha
    alpha = min(1, (q_forward / q_backward) * g_ratio)
    return alpha


def accept_or_reject_proposal(current_partition, proposed_partition, V_CP, q, g_func):
    """
    Accept or reject the proposed partition based on the computed acceptance probability.

    Parameters:
    - current_partition: Current partition π.
    - proposed_partition: Proposed partition π'.
    - V_CP: Set of connected components selected in Step 3.
    - q: Probability threshold for turning on edges.
    - g_func: Function to compute g(π), the unnormalized target distribution for sampling.

    Returns:
    - new_partition: Accepted partition (either proposed_partition or current_partition).
    """
    # Compute acceptance probability
    alpha = compute_acceptance_probability(current_partition, proposed_partition, V_CP, q, g_func)

    # Sample a uniform random number u ~ U(0,1)
    u = random.random()

    # Accept or reject
    if u <= alpha:
        return proposed_partition  # Accept the proposed partition
    else:
        return current_partition  # Reject the proposal and keep the current partition  # Reject the proposal and keep the current partition


def main():
    random.seed(42)
    data_path = "data/IA_raw_data.json"
    df = load_raw_data(data_path)

    # largest amount of adjacent nodes:
    # max(list(map(len, graph.values())))

    # Convert the 'geometry' column to shapely Polygon objects (if needed)
    df['geometry'] = df['geometry'].apply(lambda x: Polygon(x) if not isinstance(x, Polygon) else x)
    # Compute the centroid
    df['centroid'] = df['geometry'].apply(lambda geom: geom.centroid)
    df['coordinates'] = df['centroid'].apply(lambda point: [point.x, point.y])

    # Hyperparameters
    q = 0.05  # 0.05 or 0.04 (for PA) from the paper
    num_iterations = 10000  #
    lambda_param = 2

    # Run the algorithm
    samples = run_algorithm_1(df, q=q, num_iterations=num_iterations, lambda_param=lambda_param)

    # Reset the index and add a node_id column
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Precompute node-level data for the final reward calculations
    populations = df.set_index('node_id')['pop'].to_dict()
    votes_dem = df.set_index('node_id')['pre_20_dem_bid'].to_dict()
    votes_rep = df.set_index('node_id')['pre_20_rep_tru'].to_dict()

    # Print the samples and combined rewards
    for i, samp in enumerate(samples):
        r = combined_reward(samp, populations, votes_dem, votes_rep)
        if i % 1000 == 0:
            print(f"Iteration {i}, Reward: {r:.4f}")


if __name__ == "__main__":
    main()
