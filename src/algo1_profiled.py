import math
import random
from collections import defaultdict

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
from shapely.ops import unary_union

import cProfile
import pstats

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


def visualize_map_with_graph_and_geometry(G, E_on, boundary_components, V_CP, df, district_id_col,
                                          geometry_col="geometry"):
    """
    Combines the map visualization with the graph overlay, highlighting boundary and V_CP nodes.
    """
    global DISTRICT_COLOR_MAP

    # Ensure the district color map is initialized
    unique_districts = df[district_id_col].unique()
    if not DISTRICT_COLOR_MAP:
        initialize_district_color_map(unique_districts)

    # Convert df to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    gdf['color'] = gdf[district_id_col].map(DISTRICT_COLOR_MAP)

    # Set up figure
    fig, ax = plt.subplots(figsize=(15, 12))

    # Plot the map with districts
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black')

    # Create a set for quick lookup of E_on edges
    E_on_set = {(min(u, v), max(u, v)) for u, v in E_on}

    # Extract node coordinates
    coordinates = df.set_index('node_id')['coordinates'].to_dict()
    pos = {node: coordinates.get(node, (0, 0)) for node in G.nodes()}

    # Partition
    partition = df.set_index('node_id')[district_id_col].to_dict()

    # Flatten V_CP to get all V_CP nodes
    V_CP_nodes = {n for comp in V_CP for n in comp}

    # Assign colors to nodes
    node_colors = []
    for node in G.nodes():
        dist_id = partition.get(node, None)
        if dist_id is not None:
            district_color = DISTRICT_COLOR_MAP.get(dist_id, "#CCCCCC")

            # Check if node is in boundary_components
            is_boundary_node = any(node in comp for comp in boundary_components)

            if is_boundary_node:
                # Inverse color
                r, g, b, _ = mcolors.to_rgba(district_color)
                node_color = to_hex((1 - r, 1 - g, 1 - b))
            else:
                node_color = district_color
        else:
            # Default color if not in partition
            node_color = "#CCCCCC"

        node_colors.append(node_color)

    # Edge colors
    edge_colors = ['red' if (min(u, v), max(u, v)) in E_on_set else 'white' for u, v in G.edges()]
    edge_widths = [3 if (min(u, v), max(u, v)) in E_on_set else 2 for u, v in G.edges()]

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, edgecolors='black', alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')

    # Overlay V_CP nodes in red to highlight them
    if V_CP_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(V_CP_nodes), node_color='red', node_size=500, edgecolors='black',
                               alpha=1.0, ax=ax)

    # Build the legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='E_on edges'),
        Line2D([0], [0], color='white', lw=2, label='Other edges'),
        Patch(facecolor="#CCCCCC", edgecolor='black', label='Non-boundary nodes'),
        Patch(facecolor="black", edgecolor='black', label='Boundary nodes (inverse color)'),
        Patch(facecolor="red", edgecolor='black', label='V_CP nodes'),
    ]
    for district, color in DISTRICT_COLOR_MAP.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'District {district}'))

    state = df['state'].iloc[0]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Combined Map and Graph Visualization for {state} State", fontsize=16)

    plt.tight_layout(pad=1.0)
    plt.show()


def population_equality_reward(partition, populations):
    """
    Penalizes deviations from ideal district population quadratically.
    """
    districts = set(partition.values())
    district_pops = {d: 0 for d in districts}
    for node, dist in partition.items():
        district_pops[dist] += populations[node]

    total_pop = sum(populations.values())
    num_districts = len(districts)
    ideal_pop = total_pop / num_districts

    ssd_pop = 0.0
    for dist in districts:
        frac_dev = (district_pops[dist] / ideal_pop) - 1.0
        ssd_pop += frac_dev ** 2  # Quadratic penalty for population deviation

    # Transform to a positive reward using exponential
    return np.exp(-ssd_pop)  # Larger deviation leads to exponentially lower rewards


def voting_share_reward(partition, votes_dem, votes_rep):
    districts = set(partition.values())
    district_dem_votes = {d: 0 for d in districts}
    district_total_votes = {d: 0 for d in districts}

    for node, dist in partition.items():
        district_dem_votes[dist] += votes_dem[node]
        district_total_votes[dist] += (votes_dem[node] + votes_rep[node])

    ssd_vote = 0.0
    for dist in districts:
        if district_total_votes[dist] > 0:
            dem_share = district_dem_votes[dist] / district_total_votes[dist]
            diff = dem_share - 0.5
            ssd_vote += diff * diff
        # If no votes, ssd_vote doesn't increase

    # Again, use exponential transform
    return np.exp(-ssd_vote)


def compactness_reward(partition, gdf, district_id_col='district', w_len_width=1.0, w_perimeter=1.0):
    gdf = gdf.set_index('node_id')
    districts = set(partition.values())

    total_len_width_diff = 0.0
    total_perimeter = 0.0

    for d in districts:
        district_nodes = [node for node, dist_id in partition.items() if dist_id == d]
        district_geom = unary_union(gdf.loc[district_nodes, 'geometry'])

        if district_geom.is_empty:
            continue

        minx, miny, maxx, maxy = district_geom.bounds
        length = maxx - minx
        width = maxy - miny
        len_width_diff = abs(length - width)

        perimeter = district_geom.length

        total_len_width_diff += len_width_diff
        total_perimeter += perimeter

    # Use exponential transform for compactness
    compactness_metric = w_len_width * total_len_width_diff + w_perimeter * total_perimeter
    return np.exp(-compactness_metric)


def combined_reward(partition, populations, votes_dem, votes_rep, gdf, reward_w):
    pop_score = population_equality_reward(partition, populations)
    vote_score = voting_share_reward(partition, votes_dem, votes_rep)
    compact_score = compactness_reward(partition, gdf)

    # Weighted sum of positive scores
    # All components are now positive. Higher is better.
    return (reward_w['pop'] * pop_score) + (reward_w['vote'] * vote_score) + (reward_w['compact'] * compact_score)


def run_algorithm_1(df, reward_w, q, beta, num_samples, lambda_param=2, pop_deviation=0.10, compactness_deviation=0.10):
    """
    Executes the redistricting algorithm for the specified number of samples taken consecutively from the last accepted
    partition. The algorithm follows the steps outlined in the paper while using an updated constraint of geometric
    compactness for

    Parameters:
    - df: DataFrame containing the nodes and their attributes.
    - q: Probability threshold for turning on edges.
    - num_samples: Number of iterations to run the algorithm.
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (used in select_nonadjacent_components).

    Returns:
    - samples: List of partition samples after each iteration.
    """
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Precompute node-level data for reward calculations
    populations = df.set_index('node_id')['vap'].to_dict()
    votes_dem = df.set_index('node_id')['pre_20_dem_bid'].to_dict()
    votes_rep = df.set_index('node_id')['pre_20_rep_tru'].to_dict()
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

    # Define g_func here so it can access `populations` and `beta`
    def g_func(partition):
        """
        Compute g(π):
        g(π) = exp(-β * sum_over_districts |(pop(district)/ideal_pop - 1)|)
        """
        # Aggregate district populations using defaultdict
        district_pop = defaultdict(float)
        for node, dist in partition.items():
            district_pop[dist] += populations[node]

        # Precompute constants
        total_pop = sum(populations.values())
        num_districts = len(district_pop)
        ideal_pop = total_pop / num_districts

        # Compute deviation sum
        deviation_sum = sum(abs((pop / ideal_pop) - 1) for pop in district_pop.values())

        # Return g(π)
        return np.exp(-beta * deviation_sum)

    # Initialize the partition and store initial results for visualization
    current_partition = initial_partition.copy()

    # For visualization
    district = np.array([current_partition[node] for node in range(len(df))])
    df['district'] = district

    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # List to store partition samples
    samples = []
    i = 0

    # Initialize best partition and reward
    best_partition = current_partition.copy()
    best_reward = combined_reward(current_partition, populations, votes_dem, votes_rep, gdf, reward_w)

    # Iterative algorithm
    while i < num_samples:
        # Step 1: Determine E_on edges (Select edges in the same district with probability q)
        E_on = turn_on_edges(G, current_partition, q)

        # Step 2: Find boundary components (connected components with neighbors in different districts)
        boundary_components = find_boundary_connected_components(G, current_partition, E_on)

        # Step 3: Select a subgroup of nonadjacent components along boundaries that will get swapped
        V_CP = select_nonadjacent_components(boundary_components, G, current_partition, lambda_param=lambda_param)

        # Visualization before changes (optional; can slow down execution if num_samples is large)
        # visualize_map_with_graph_and_geometry(G, E_on, boundary_components, V_CP, df, 'district')

        # Step 4: Propose swaps for the selected components (random order => can cancel each other out)
        proposed_partition = propose_swaps(current_partition, V_CP, G)

        # Step 5: Hard check for equal population constraint + geometry compactness (and maybe voting share balance later)
        max_pop_dev = max_population_deviation(proposed_partition, populations)
        avg_compactness = max_compactness_deviation(proposed_partition, gdf)
        # avg_compactness = average_compactness_deviation(proposed_partition, gdf)
        if max_pop_dev > pop_deviation or avg_compactness > compactness_deviation:
            continue

        # Polsby-Popper score for compactness: 4 * pi * area / perimeter^2

        # Step 6: Accept or reject the proposed partition based on the acceptance probability (Metropolis-Hastings)
        if proposed_partition != current_partition and accept_or_reject_proposal(current_partition, proposed_partition,
                                                                                 V_CP, q, g_func):
            current_partition = proposed_partition
            proposed_reward = combined_reward(current_partition, populations, votes_dem, votes_rep, gdf, reward_w)

            # Update the DataFrame with the new partition for visualization
            district = np.array([proposed_partition[node] for node in range(len(df))])
            df['district'] = district

            if proposed_reward > best_reward:
                best_reward = proposed_reward
                best_partition = proposed_partition
                print(
                    f"Iteration {i} | Avg Compacness: {avg_compactness:.6f} | New Best Reward: {proposed_reward:.6f} !!!")
            else:
                print(f"Iteration {i} | Avg Compacness: {avg_compactness:.6f} | Proposed Reward: {proposed_reward:.6f}")

            # Save the current partition
            samples.append(current_partition.copy())
            i += 1

    return samples, best_partition


def max_population_deviation(partition, populations):
    """
    Compute the maximum absolute deviation of district populations from the ideal population.

    Parameters:
    - partition: dict mapping node -> district_id
    - populations: dict mapping node -> population

    Returns:
    - max_dev: A float representing the maximum absolute fractional deviation from the ideal population.
    """
    # Aggregate district populations
    district_pop = defaultdict(float)
    for node, dist in partition.items():
        district_pop[dist] += populations[node]

    # Precompute constants
    total_pop = sum(populations.values())
    num_districts = len(district_pop)  # Equivalent to len(set(partition.values()))
    ideal_pop = total_pop / num_districts

    # Compute maximum fractional deviation
    max_dev = max(abs((p / ideal_pop) - 1) for p in district_pop.values())

    return max_dev


def max_compactness_deviation(partition, gdf):
    """
    Compute the maximum compactness deviation as a percentage.

    For each district:
    - Length-width deviation: How far the bounding box differs from a square, as a fraction (0 to 1).
      Multiply by 100 for a percentage.
    - Perimeter deviation: Compare the district's perimeter to that of a circle with the same area.
      (perimeter/circle_perimeter - 1)*100 gives the percentage increase.

    We then combine them by averaging to get a compactness percentage deviation.
    Finally, return the maximum such percentage among all districts.
    """

    if gdf.index.name != 'node_id':
        gdf = gdf.set_index('node_id')

    districts = set(partition.values())
    deviations = []

    for d in districts:
        district_nodes = [node for node, dist_id in partition.items() if dist_id == d]
        if not district_nodes:
            continue

        district_geom = unary_union(gdf.loc[district_nodes, 'geometry'])

        if district_geom.is_empty:
            continue

        # Compute bounding box
        minx, miny, maxx, maxy = district_geom.bounds
        length = maxx - minx
        width = maxy - miny

        # Length-width deviation fraction
        if length == 0 or width == 0:
            # Essentially a line, 100% deviation
            len_width_dev_percent = 100.0
        else:
            len_width_dev = abs(length - width) / max(length, width)
            len_width_dev_percent = len_width_dev * 100.0

        # Perimeter deviation relative to a circle
        district_area = district_geom.area
        district_perimeter = district_geom.length

        if district_area > 0:
            # Ideal circle perimeter for same area: P = 2 * sqrt(pi * A)
            circle_perimeter = 2.0 * math.sqrt(math.pi * district_area)
            perimeter_ratio = district_perimeter / circle_perimeter
            # Percentage increase in perimeter compared to circle
            perimeter_dev_percent = (perimeter_ratio - 1.0) * 100.0
        else:
            # If no area, treat as maximal deviation
            perimeter_dev_percent = 100.0

        # Combine the two percentages
        # Here we simply average them, but you can weight them differently if you prefer
        district_deviation_percent = (len_width_dev_percent + perimeter_dev_percent) / 2.0

        deviations.append(district_deviation_percent)

    max_dev = max(deviations) if deviations else 0.0
    return max_dev


def average_compactness_deviation(partition, gdf):
    """
    Compute the average compactness deviation as a percentage.

    For each district:
    - Length-width deviation: How far the bounding box differs from a square, as a fraction (0 to 1).
      Multiply by 100 for a percentage.
    - Perimeter deviation: Compare the district's perimeter to that of a circle with the same area.
      (perimeter/circle_perimeter - 1)*100 gives the percentage increase.

    Combine these two percentages for each district and return their average.
    """
    # Ensure gdf is indexed by 'node_id'
    if gdf.index.name != 'node_id':
        gdf = gdf.set_index('node_id')

    # Precompute districts and group nodes
    district_nodes = {}
    for node, dist_id in partition.items():
        if dist_id not in district_nodes:
            district_nodes[dist_id] = []
        district_nodes[dist_id].append(node)

    deviations = []

    for district, nodes in district_nodes.items():
        # Skip empty districts
        if not nodes:
            continue

        # Combine geometries for the district
        district_geom = unary_union(gdf.loc[nodes, 'geometry'])

        if district_geom.is_empty:
            continue

        # Compute bounding box
        minx, miny, maxx, maxy = district_geom.bounds
        length = maxx - minx
        width = maxy - miny

        # Length-width deviation fraction
        len_width_dev_percent = (
            100.0 if length == 0 or width == 0
            else abs(length - width) / max(length, width) * 100.0
        )

        # Perimeter deviation relative to a circle
        district_area = district_geom.area
        district_perimeter = district_geom.length
        if district_area > 0:
            circle_perimeter = 2.0 * math.sqrt(math.pi * district_area)
            perimeter_dev_percent = (district_perimeter / circle_perimeter - 1.0) * 100.0
        else:
            perimeter_dev_percent = 100.0

        # Combine the two percentages
        district_deviation_percent = (len_width_dev_percent + perimeter_dev_percent) / 2.0
        deviations.append(district_deviation_percent)

    # Compute average deviation
    average_dev = sum(deviations) / len(deviations) if deviations else 0.0
    return average_dev


def turn_on_edges(G, partition, q):
    """
    Turns on edges that are in the same district with probability q.
    Used to make random connected components in the graph, or small sub-graphs inside a district.
    """
    return [
        (u, v)
        for u, v in G.edges()
        if partition[u] == partition[v] and random.random() < q
    ]


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
        C = set(random.sample(list(remaining_components), 1)[0])
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
    # Ensure component is a set
    component = set(component)

    # Determine the current district
    current_district = next(iter({partition[n] for n in component}))

    # Find all neighbors of nodes in the component
    neighbors = set(
        neighbor for node in component for neighbor in G.neighbors(node)
    )

    # Exclude nodes within the component
    external_neighbors = neighbors - component

    # Collect districts of external neighbors not in the current district
    adjacent_districts = {partition[neighbor] for neighbor in external_neighbors
                          if partition[neighbor] != current_district}

    return list(adjacent_districts)


def accept_or_reject_proposal(current_partition, proposed_partition, V_CP, q, g_func):
    """
    A simple Metropolis-Hastings accept/reject step.

    Parameters:
    - current_partition: The current partition π.
    - proposed_partition: The proposed new partition π'.
    - V_CP: The set of selected boundary components (not used here, but kept for consistency).
    - q: Probability threshold for turning on edges (not used directly here).
    - g_func: A function that returns the unnormalized target probability g(π).

    Returns:
    - A partition (dict): Either the proposed_partition if accepted, or the current_partition if rejected.
    """
    g_current = g_func(current_partition)
    g_proposed = g_func(proposed_partition)

    # Compute the Metropolis-Hastings acceptance ratio
    # Assuming symmetric proposals, ratio = g(proposed)/g(current).
    if g_current == 0:
        # If g_current is zero and g_proposed > 0, always accept.
        # Otherwise, if both zero, just reject.
        if g_proposed > 0:
            ratio = 0
        else:
            ratio = 1
    else:
        ratio = g_proposed / g_current

    # Acceptance probability α = min(1, ratio)
    alpha = min(1.0, ratio)

    # Draw a uniform random number to decide acceptance
    u = random.random()

    return u <= alpha


def main():
    random.seed(42)
    data_path = "../data/IA_raw_data.json"
    data_path = "data/IA_raw_data.json"
    df = load_raw_data(data_path)

    # Convert the 'geometry' column to shapely Polygon objects (if needed)
    df['geometry'] = df['geometry'].apply(lambda x: Polygon(x) if not isinstance(x, Polygon) else x)
    df['coordinates'] = df['geometry'].apply(lambda geom: geom.centroid)  # Compute the centroid of each polygon
    df['coordinates'] = df['coordinates'].apply(lambda point: [point.x, point.y])

    reward_w = {
        'pop': 0.50,
        'vote': 0.80,
        'compact': 0.30
    }

    # Run the algorithm with current parameters
    samples, best_partition = run_algorithm_1(df,  # DataFrame containing the nodes and their attributes
                                              reward_w,  # Reward weights
                                              q=0.05,  # 0.05 or 0.04 (for PA) from the paper
                                              beta=40,  # Inverse temperature parameter
                                              num_samples=200,  # Number of samples to generate (not total iterations)
                                              lambda_param=2,  # Lambda parameter for zero-truncated Poisson dist
                                              pop_deviation=0.07,  # Population deviation
                                              compactness_deviation=85)  # Compactness deviation

    # Reset the index and add a node_id column
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Update the DataFrame with the best partition
    district = np.array([best_partition[node] for node in range(len(df))])
    df['district'] = district
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Precompute node-level data for reward calculations
    populations = df.set_index('node_id')['pop'].to_dict()
    votes_dem = df.set_index('node_id')['pre_20_dem_bid'].to_dict()
    votes_rep = df.set_index('node_id')['pre_20_rep_tru'].to_dict()

    best_reward = combined_reward(best_partition, populations, votes_dem, votes_rep, gdf, reward_w)
    print(f"Current Reward: {best_reward:.4f}")

    # Visualize the best partition
    metrics = {
        "total": [("vap", "Voting Age Population"), ("pop", "Total Population")],
        "mean": [],
        "ratio": [[("pre_20_dem_bid", "Biden"), ("pre_20_rep_tru", "Trump")]]
    }
    visualize_map_with_geometry(df, geometry_col="geometry", district_id_col="district", state="Iowa", metrics=metrics)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.runcall(main)
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    stats.print_stats("algo1_profiled.py")
    stats.dump_stats("profile_results.pstats")
