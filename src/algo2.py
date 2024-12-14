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

# # Step 5: Evaluate rewards
# current_reward = combined_reward(current_partition, populations, votes_dem, votes_rep)
# proposed_reward = combined_reward(proposed_partition, populations, votes_dem, votes_rep)
#
# # Step 6: Accept or reject the proposed partition
# if proposed_reward >= current_reward:
#     current_partition = proposed_partition


def run_algorithm_1(df, q, beta, num_iterations, lambda_param=2):
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

        # vis before update
        visualize_map_with_graph_and_geometry(G, E_on, boundary_components, df, 'district')

        # Step 5: Accept or reject the proposed partition
        current_partition = accept_or_reject_proposal(
            G, current_partition, proposed_partition, V_CP, g_func
        )

        # Step 7: Save the current partition
        samples.append(current_partition.copy())

        # check if the partition has changed to the new one
        if current_partition == proposed_partition:
            print(f"Accepted: {current_partition} -> {proposed_partition}")

            # Update the DataFrame with the new partition for visualization
            district = np.array([proposed_partition[node] for node in range(len(df))])
            df['district'] = district
        # Visualization (optional; can slow down execution if num_iterations is large)
        # visualize_map_with_graph_and_geometry(G, E_on, boundary_components, df, 'district')

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


def g_beta(partition, populations, beta):
    """
    Compute g_beta(pi):
    g_beta(pi) = exp(-beta * sum_over_districts |(pop(district)/ideal_pop) - 1|)
    """
    # Compute district populations
    district_pop = {}
    for node, dist in partition.items():
        district_pop[dist] = district_pop.get(dist, 0) + populations[node]

    total_pop = sum(populations.values())
    num_districts = len(set(partition.values()))
    ideal_pop = total_pop / num_districts

    # Compute sum of absolute deviations
    deviation_sum = 0.0
    for dist, p in district_pop.items():
        deviation_sum += abs((p / ideal_pop) - 1)

    return np.exp(-beta * deviation_sum)


def truncated_poisson_pmf(k, lam):
    """
    PMF of a zero-truncated Poisson distribution:
    P(R=k) = (Poisson(k; lam)) / (1 - Poisson(0; lam))
    for k >= 1
    """
    # Poisson(0; lam) = exp(-lam)
    return (np.exp(-lam)*lam**k / np.math.factorial(k)) / (1 - np.exp(-lam))

def truncated_poisson_cdf(k, lam):
    """
    CDF of zero-truncated Poisson: F(k) = P(R <= k)
    = sum_{j=1 to k} truncated_poisson_pmf(j, lam)
    """
    return sum(truncated_poisson_pmf(j, lam) for j in range(1, k+1))

def count_boundary_components(G, partition, E_on):
    """
    Count the number of boundary components: |B(CP, π)|
    This is basically the length of find_boundary_connected_components(G, partition, E_on).
    """
    bc = find_boundary_connected_components(G, partition, E_on)
    return len(bc)


def count_cut_edges(G, partition, V_CP):
    """
    Count |C(π, V_CP)|, the number of edges that would be cut by removing V_CP.
    This can be interpreted as the number of edges (u,v) in G where u in V_CP and v not in V_CP,
    or vice versa.
    """
    V_CP_set = set(V_CP)  # Flatten if V_CP is a list of tuples
    # If V_CP is a list of components (each component is a tuple), flatten them:
    if any(isinstance(el, (tuple, list, set)) for el in V_CP):
        flat = set()
        for comp in V_CP:
            flat.update(comp)
        V_CP_set = flat

    cut_count = 0
    for u in V_CP_set:
        for v in G.neighbors(u):
            if v not in V_CP_set and partition[v] != partition[u]:
                # This edge crosses a district boundary after removing V_CP nodes
                cut_count += 1
    return cut_count


def compute_acceptance_probability(G, current_partition, proposed_partition, V_CP, q, beta, lam, E_on):
    """
    Compute α(π', CP | π) using the given formula.

    α(π',CP|π) ≈ min( 1,
       (|B(CP, π')|^R / |B(CP, π)|^R) * (F(|B(CP, π')|)/F(|B(CP, π)|)) *
       ((1-q)^{|C(π,V_CP)|}/(1-q)^{|C(π',V_CP)|}) * (g_beta(π')/g_beta(π))
    )

    We need R and F(), which depend on the chosen R from the truncated Poisson.
    In your code, you must have chosen R in select_nonadjacent_components.
    Make sure to return R from that step or store it somewhere accessible.
    """

    # Retrieve populations from G or a global structure
    # Assume we have populations as a global dict or pass it as a parameter
    # For demonstration, assume populations is accessible:
    # You may need to store populations globally or pass them in.
    populations = nx.get_node_attributes(G, 'pop')
    if not populations:
        raise ValueError("Node populations not found in the graph attributes.")

    # Compute g_beta(π) and g_beta(π')
    g_curr = g_beta(current_partition, populations, beta)
    g_prop = g_beta(proposed_partition, populations, beta)

    # Compute |B(CP, π)| and |B(CP, π')|
    BCP_curr = count_boundary_components(G, current_partition, E_on)
    BCP_prop = count_boundary_components(G, proposed_partition, E_on)  # Need E_on for π'?
    # In practice, the E_on edges are determined for the current step. For the proposed partition,
    # E_on might be the same since q-step is done before proposal.
    # If necessary, recompute E_on for π' if the algorithm requires it.

    # Retrieve R and F(.)
    # You must ensure that the function select_nonadjacent_components returns R or store it globally.
    # Suppose we stored R as a global variable or returned it alongside V_CP.
    # Similarly, we must have chosen R from a truncated Poisson at step 3.
    R = get_current_R()  # This is a placeholder. Make sure to return R from the selection step.

    F_BCP_curr = truncated_poisson_cdf(BCP_curr, lam)
    F_BCP_prop = truncated_poisson_cdf(BCP_prop, lam)

    # Compute |C(π,V_CP)| and |C(π',V_CP)|
    C_curr = count_cut_edges(G, current_partition, V_CP)
    C_prop = count_cut_edges(G, proposed_partition, V_CP)

    # Compute acceptance ratio
    # Handle cases where denominators are zero or F(...) = 0
    if BCP_curr == 0 or BCP_prop == 0 or F_BCP_curr == 0 or F_BCP_prop == 0:
        # If these are zero, proposal might be invalid. You could directly reject or handle carefully.
        return 0.0

    ratio = ((BCP_prop**R) / (BCP_curr**R)) * (F_BCP_prop / F_BCP_curr) * ((1-q)**C_curr / (1-q)**C_prop) * (g_prop / g_curr)

    alpha = min(1.0, ratio)
    return alpha

def accept_or_reject_proposal(G, current_partition, proposed_partition, V_CP, q, beta, lam, E_on):
    """
    Accept or reject the proposed partition based on the computed acceptance probability.
    """
    alpha = compute_acceptance_probability(G, current_partition, proposed_partition, V_CP, q, beta, lam, E_on)
    u = random.random()
    if u <= alpha:
        return proposed_partition
    else:
        return current_partition


def run_algorithm_1_2(df, q, M, S, delta, beta, lam=2):
    """
    Implements Algorithm 1.2:
    1. Run Algorithm 1 (with Gibbs) for M steps to generate M samples.
    2. Filter samples by population constraint δ.
    3. Resample S samples via SIR using weights 1/g_beta(π).

    Assumes run_algorithm_1 returns M samples.

    Parameters:
    - df: DataFrame with node attributes including 'pop'.
    - q: Probability of turning on edges for Algorithm 1.
    - M: Number of samples to generate from Algorithm 1.
    - S: Number of samples to resample in Step 3.
    - delta: Population constraint threshold.
    - beta: Gibbs parameter β.
    - lam: Parameter for zero-truncated Poisson (used in selection of R).

    Returns:
    - A list of S resampled partitions.
    """

    # Run Algorithm 1 (you have a defined function run_algorithm_1)
    # Make sure run_algorithm_1 uses accept_or_reject_proposal as defined above or adapted accordingly.
    samples = run_algorithm_1(df, q, num_iterations=M, lambda_param=lam)

    df = df.reset_index(drop=True)
    df['node_id'] = df.index
    populations = df.set_index('node_id')['pop'].to_dict()

    # Filter samples by population constraint
    filtered_samples = []
    filtered_weights = []
    for sample in samples:
        if check_population_constraint(sample, populations, delta):
            g_val = g_beta(sample, populations, beta)
            if g_val > 0:
                filtered_samples.append(sample)
                filtered_weights.append(1.0 / g_val)

    if len(filtered_samples) == 0:
        print("No samples met the population constraint. Try adjusting δ or M.")
        return []

    # SIR Resampling
    weights = np.array(filtered_weights)
    weights /= weights.sum()
    indices = np.random.choice(len(filtered_samples), size=S, replace=True, p=weights)
    resampled_samples = [filtered_samples[i] for i in indices]

    return resampled_samples

#
# def q_func(G, current_partition, proposed_partition, V_CP):
#     """
#     Compute the transition probability q(π', CP | π) or q(π, CP | π').
#
#     Parameters:
#     - current_partition: Current partition π.
#     - proposed_partition: Proposed partition π'.
#     - V_CP: Set of connected components selected in Step 3.
#
#     Returns:
#     - q_value: Transition probability.
#     """
#     # Probability of selecting the specific connected components in V_CP
#     num_boundary_components = len(V_CP)
#     q_vcp = 1.0 / max(1, num_boundary_components)  # Uniform selection probability
#
#     # Probability of assigning each component to a new district
#     reassignment_prob = 1.0
#     for component in V_CP:
#         current_district = {current_partition[node] for node in component}.pop()
#         neighbor_districts = find_neighboring_districts(component, proposed_partition, G=G)  # Update G if needed
#         if neighbor_districts:
#             reassignment_prob *= 1.0 / len(neighbor_districts)
#         else:
#             reassignment_prob = 0.0  # If no valid reassignments, the probability is 0
#
#     return q_vcp * reassignment_prob
#
#
# def compute_acceptance_probability(G, current_partition, proposed_partition, V_CP, g_func):
#     """
#     Computes the acceptance probability α(π', CP | π).
#
#     Parameters:
#     - current_partition: Current partition π (dict mapping node to district).
#     - proposed_partition: Proposed partition π'.
#     - V_CP: Set of connected components selected in Step 3.
#     - q_func: Function to compute q(π, CP | π') or q(π', CP | π).
#     - g_func: Function to compute g(π), the unnormalized target distribution for sampling.
#
#     Returns:
#     - alpha: Acceptance probability for the proposal.
#     """
#     # Step 1: Compute q(π', CP | π) and q(π, CP | π')
#     q_forward = q_func(G, current_partition, proposed_partition, V_CP)  # Probability of transitioning forward
#     q_backward = q_func(G, proposed_partition, current_partition, V_CP)  # Probability of transitioning backward
#
#     # Step 2: Compute g(π') / g(π)
#     g_current = g_func(current_partition)
#     g_proposed = g_func(proposed_partition)
#
#     if g_current == 0:
#         if g_proposed > 0:
#             return 1.0  # Accept with probability 1 if the current g(π) is zero but the proposed is valid
#         else:
#             return 0.0  # Reject if both g(π) and g(π') are zero
#
#     g_ratio = g_proposed / g_current
#
#     # Step 3: Compute acceptance probability α
#     alpha = min(1, (q_forward / q_backward) * g_ratio)
#     return alpha
#
#
# def accept_or_reject_proposal(G, current_partition, proposed_partition, V_CP, g_func):
#     """
#     Accept or reject the proposed partition based on the computed acceptance probability.
#
#     Parameters:
#     - current_partition: Current partition π.
#     - proposed_partition: Proposed partition π'.
#     - V_CP: Set of connected components selected in Step 3.
#     - q: Probability threshold for turning on edges.
#     - g_func: Function to compute g(π), the unnormalized target distribution for sampling.
#
#     Returns:
#     - new_partition: Accepted partition (either proposed_partition or current_partition).
#     """
#     # Compute acceptance probability
#     alpha = compute_acceptance_probability(G, current_partition, proposed_partition, V_CP, g_func)
#
#     # Sample a uniform random number u ~ U(0,1)
#     u = random.random()
#
#     # Accept or reject
#     if u <= alpha:
#         return proposed_partition  # Accept the proposed partition
#     else:
#         return current_partition  # Reject the proposal and keep the current partition  # Reject the proposal and keep the current partition


def check_population_constraint(partition, populations, delta):
    """
    Check if a given partition π satisfies the population constraint δ.
    The constraint is typically of the form:
    max_{district} (district_population / ideal_population) <= δ

    Parameters:
    - partition: dict mapping node_id -> district_id
    - populations: dict mapping node_id -> population
    - delta: The population constraint threshold

    Returns:
    - Boolean indicating if the partition meets the population constraint.
    """
    # Compute number of districts and their populations
    district_pop = {}
    for node, dist in partition.items():
        district_pop[dist] = district_pop.get(dist, 0) + populations[node]

    # Ideal population
    total_pop = sum(populations.values())
    num_districts = len(set(partition.values()))
    ideal_pop = total_pop / num_districts

    # Check constraint
    max_ratio = max(p / ideal_pop for p in district_pop.values())
    return max_ratio <= delta

def sir_resampling(samples, weights, S):
    """
    Perform Sampling/Importance Resampling (SIR) given a set of samples and their weights.

    Parameters:
    - samples: List of partition samples
    - weights: List of weights associated with each sample (unnormalized)
    - S: Number of samples to draw with replacement

    Returns:
    - A list of S resampled partitions.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    indices = np.random.choice(len(samples), size=S, replace=True, p=weights)
    return [samples[i] for i in indices]


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
    q = 0.05  # edge cut probability (0.05 or 0.04 from the paper)
    beta = 10.0  # Inverse temperature parameter
    num_iterations = 100 # Number of total iterations f
    lambda_param = 2  # Lambda parameter for zero-truncated Poisson distribution
    M = 100  # Number of samples to generate from Algorithm 1
    S = 10  # Number of samples to resample via SIR
    pop_deviation = 0.10  # Population deviation threshold

    # Run the algorithm
    samples = run_algorithm_1(df, q=q, beta=beta, num_iterations=num_iterations, lambda_param=lambda_param)

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
