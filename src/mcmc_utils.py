import argparse
import math
import random
import time
from collections import defaultdict
from typing import Optional
from scipy.stats import poisson
from itertools import chain
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
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from dataclasses import dataclass

import cProfile
import pstats

from utils.data_utils import *

STATE_ABBREVIATIONS = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia"
}

# Create a persistent district-to-color mapping
DISTRICT_COLOR_MAP = {}

# Global dictionary for node sizes based on state abbreviations
NODE_SIZES = {
    'IA': 700,
    'PA': 50,
    # Add more states and their sizes as needed
}


# Define a dataclass for Hyperparameters
@dataclass
class Hyperparameters:
    q: float
    beta: Optional[int]  # int or None
    num_iterations: int
    M: Optional[int]  # int or None
    S: Optional[int]  # int or None
    lambda_param: float
    max_pop_dev_threshold: float
    compactness_threshold: float
    compactness_constraint: bool


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

    # If the district_id_col is not a string, add a new column to the DataFrame called 'district' with the values
    if not isinstance(district_id_col, str):
        df['district'] = district_id_col
        district_id_col = 'district'

    # Ensure the district color map is initialized
    unique_districts = df[district_id_col].unique()
    if not DISTRICT_COLOR_MAP:
        initialize_district_color_map(unique_districts)

    # Convert df to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    gdf['color'] = gdf[district_id_col].map(DISTRICT_COLOR_MAP)

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # More compact with high DPI

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

    # Assign colors and sizes to nodes
    node_colors = []
    node_sizes = []
    df_indexed = df.set_index('node_id')
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

        # Get state abbreviation for the node
        if node in df_indexed.index:
            state_abbr = df_indexed.loc[node, 'state']
            if isinstance(state_abbr, (list, tuple)):
                state_abbr = state_abbr[0]
            node_size = NODE_SIZES.get(state_abbr, 700)  # Default size if state is not in NODE_SIZES
        else:
            node_size = 700  # Default size

        # Make `V_CP` nodes significantly larger
        if node in V_CP_nodes:
            node_size *= 1.0  # Scale up for V_CP nodes

        node_sizes.append(node_size)

    # Edge colors and widths
    edge_colors = ['red' if (min(u, v), max(u, v)) in E_on_set else 'white' for u, v in G.edges()]
    edge_widths = [4 if (min(u, v), max(u, v)) in E_on_set else 3 for u, v in G.edges()]

    # Highlight edges connected to `V_CP` nodes
    for idx, (u, v) in enumerate(G.edges()):
        if u in V_CP_nodes or v in V_CP_nodes:
            edge_widths[idx] = 6  # Larger width for V_CP edges

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors='black', alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')

    # Overlay `V_CP` nodes in red to highlight them, adjust sizes
    if V_CP_nodes:
        V_CP_node_sizes = []
        for node in V_CP_nodes:
            if node in df_indexed.index:
                state_abbr = df_indexed.loc[node, 'state']
                if isinstance(state_abbr, (list, tuple)):
                    state_abbr = state_abbr[0]
                node_size = NODE_SIZES.get(state_abbr, 100) * 1.6  # Make V_CP nodes larger
            else:
                node_size = 200  # Default larger size for V_CP nodes
            V_CP_node_sizes.append(node_size)

        nx.draw_networkx_nodes(G, pos, nodelist=list(V_CP_nodes), node_color='red', node_size=V_CP_node_sizes,
                               edgecolors='black', alpha=1.0, ax=ax)

    # Build the legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=6, label='V_CP edges (highlighted)'),  # V_CP edges
        Patch(facecolor="white", edgecolor='black', label='Other edges'),
        Patch(facecolor="#CCCCCC", edgecolor='black', label='Non-boundary nodes'),
        Patch(facecolor="black", edgecolor='black', label='Boundary nodes (inverse color)'),
        Patch(facecolor="red", edgecolor='black', label='V_CP nodes')
    ]
    for district, color in DISTRICT_COLOR_MAP.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'District {district}'))

    # Extract the state abbreviation for the title
    state = df['state'].iloc[0]
    if isinstance(state, (list, tuple)):
        state = state[0]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Combined Map and Graph Visualization for {STATE_ABBREVIATIONS[state]} State", fontsize=16)

    plt.tight_layout(pad=1.0)
    plt.savefig("zoomed_in_map.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def compute_avg_compactness(partition, gdf):
    """
    Optimized compactness calculation for districts using the Polsby-Popper compactness score.
    Can also change this to max to avoid extreme outliers

    Parameters:
        partition (dict): Mapping of node IDs to districts.
        gdf (GeoDataFrame): GeoDataFrame containing geometry and node_id.

    Returns:
        float: Average compactness score.
    """
    # Add district column in one step
    gdf['district'] = gdf['node_id'].map(partition)

    # Group geometries by district and compute unary union
    district_geoms = (
        gdf.groupby('district')['geometry']
        .apply(lambda x: unary_union(x))
        .reset_index(drop=True)
    )

    # Precompute constants
    four_pi = 4 * np.pi

    # Calculate compactness for all districts in a single step
    compactness = [
        (four_pi * area) / (length ** 2)
        for geom in district_geoms
        if (area := geom.area) > 0 and (length := geom.length) > 0
    ]

    # Return the mean compactness, avoiding division by zero
    return np.mean(compactness) if compactness else 0.0


def max_population_deviation(partition, populations, total_pop=None, ideal_pop=None):
    """
        Compute the maximum absolute deviation of district populations from the ideal population.

        Parameters:
        - partition: dict mapping node -> district_id
        - populations: dict mapping node -> population
        - total_pop: Total population of the state (optional).
        - ideal_pop: Ideal population per district (optional).

        Returns:
        - max_dev: A float representing the maximum absolute fractional deviation from the ideal population.
    """
    if total_pop is None:
        total_pop = sum(populations.values())
    if ideal_pop is None:
        ideal_pop = total_pop / len(set(partition.values()))

    district_pop = defaultdict(float)
    for node, dist in partition.items():
        district_pop[dist] += populations[node]

    return max(abs((p / ideal_pop) - 1) for p in district_pop.values())


def turn_on_edges(G, partition, q):
    """
    Turns on edges that are in the same district with probability q.
    Used to make random connected components in the graph, or small sub-graphs inside a district.
    """
    edges = G.edges
    return [(u, v) for u, v in edges if partition[u] == partition[v] and random.random() < q]


def find_boundary_connected_components(G, partition, E_on):
    """
    Identify boundary components based on the turned on (E_on) edges.
    Components are valid if at least one node in the component has a neighbor in a different district.
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
    """
    Identify boundary nodes based on the current partition.
    A node is a boundary node if it has a neighbor in a different district.

    Parameters:
    - graph: The graph (NetworkX object).
    - partition: Current district assignments.

    Returns:
    - A set of boundary nodes.
    """
    boundary_nodes = set()
    for node in graph:
        if any(partition[neigh] != partition[node] for neigh in chain(graph[node])):
            boundary_nodes.add(node)
    return boundary_nodes


def is_component_on_boundary(component, partition, G):
    """
    Check if a component has any node with a neighbor in a different district.
    """
    component_district = {partition[node] for node in component}
    if len(component_district) != 1:
        return False  # The component spans multiple districts, which is invalid

    component_district = component_district.pop()
    neighbors = {neigh for node in component for neigh in G.neighbors(node)}
    if any(neigh not in component and partition[neigh] != component_district for neigh in neighbors):
        return True  # At least one neighbor is in a different district


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
    R is the number of connected components to select from the boundary.

    1) Randomly sample R from a zero-truncated Poisson distribution and select R components.
    2) continue sampling connected components until |V_CP| = R or until there are no more components left in B(CP, π)
    while ensuring they are nonadjacent and do not cause noncontiguous districts.

    Parameters:
    - boundary_components: List of connected components along the boundary.
    - G: The graph (NetworkX object).
    - partition: Current district assignments.
    - lambda_param: Lambda for the zero-truncated Poisson distribution.

    Returns:
    - V_CP: Selected nonadjacent connected components.
    """
    if len(boundary_components) == 0:
        return [], 0

    # Precompute district nodes
    district_nodes = precompute_district_nodes(partition, G)

    # Step 3(a): Generate R from a zero-truncated Poisson distribution
    max_components = len(boundary_components)
    R = 0
    # Sample R from a zero-truncated Poisson(λ) distribution capped at max_components
    while R < 1 or R > max_components:
        R = poisson.rvs(lambda_param)

    # Step 3(b): Initialize V_CP and start selecting components
    V_CP = set()
    V_CP_list = []
    remaining_components = {frozenset(component) for component in boundary_components}

    while len(V_CP_list) < R and remaining_components:
        C = set(random.sample(list(remaining_components), 1)[0])
        remaining_components.remove(frozenset(C))

        # Check if C is adjacent to any component in V_CP or causes noncontiguous districts
        if not is_adjacent_to_vcp(C, V_CP, G) and not causes_noncontiguous_district(V_CP | C, district_nodes, G):
            V_CP |= C  # Add C to V_CP only if it satisfies all constraints
            V_CP_list.append(tuple(C))

    # V_CP_list might be less than R if we couldn't pick enough
    # But R is what we tried to pick. The acceptance formula uses this R.
    # If we picked fewer, R is still what was initially chosen (this aligns with the method in the paper)
    return V_CP_list, R


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
    Check if the removal of V_CP causes a noncontiguous district (creates a new districts by breaking one into two).

    Parameters:
    - V_CP: Set of selected components to be removed.
    - district_nodes: Precomputed mapping of districts to their nodes.
    - G: The graph (NetworkX object).

    Returns:
    - Boolean indicating if removing V_CP causes a noncontiguous district.
    """
    if not V_CP:
        return False
    district_subgraphs = {d: G.subgraph(nodes - V_CP) for d, nodes in district_nodes.items() if nodes - V_CP}
    return any(not nx.is_connected(subgraph) for subgraph in district_subgraphs.values())


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


def truncated_poisson_pmf(r, lambda_, max_count):
    """
    Computes P(R=r | 1 ≤ R ≤ max_count) for a Poisson(λ) random variable truncated
    between 1 and max_count.
    """
    if r < 1 or r > max_count:
        return 0.0
    # Compute denominator: sum of Poisson(k; λ) for k=1 to max_count
    denom = poisson.cdf(max_count, lambda_) - poisson.cdf(0, lambda_)
    return poisson.pmf(r, lambda_) / denom


def count_components_in_vcp_under_partition(G, partition, q, V_CP_list):
    """
    Given a partition π', compute |C(π', V_CP)|, i.e., how many connected components
    the set of nodes in V_CP forms when edges are turned on according to π'.

    Parameters:
    - G: The graph
    - partition: Proposed partition π'
    - q: Probability for turning on edges
    - V_CP_list: List of chosen components from π (list of tuples of nodes)

    Returns:
    - An integer representing |C(π', V_CP)|
    """
    if not V_CP_list:
        return 0

    # Combine all chosen components into a set of nodes
    V_CP_nodes = set()
    for comp in V_CP_list:
        V_CP_nodes.update(comp)

    # Turn on edges under π'
    E_on_proposed = turn_on_edges(G, partition, q)

    # Subgraph induced by V_CP_nodes with E_on_proposed edges
    E_on_set = {(min(u, v), max(u, v)) for u, v in E_on_proposed}
    V_CP_subgraph = nx.Graph()
    V_CP_subgraph.add_nodes_from(V_CP_nodes)
    for u, v in E_on_set:
        if u in V_CP_nodes and v in V_CP_nodes:
            V_CP_subgraph.add_edge(u, v)

    # Count connected components in this induced subgraph
    components = nx.connected_components(V_CP_subgraph)
    count = sum(1 for _ in components)
    return count


def compute_swendsen_wang_cut(G, partition, V_CP_list):
    """
    Compute |C(π, V_CP)|: The number of edges that cut off V_CP components from the rest of the graph,
    respecting the current partition π.

    The Swendsen-Wang cut is crucial in the Metropolis-Hastings acceptance ratio because it helps quantify how V_CP
    components interact with the rest of the graph.

    These edges are relevant because they cross district boundaries under the current partition and the number edges
    that crosses the boundary with the proposed partition will change.

    Parameters:
    - G: The graph.
    - partition: Current partition π (node -> district mapping).
    - V_CP_list: List of chosen boundary components V_CP.

    Returns:
    - Integer representing the size of the Swendsen-Wang cut |C(π, V_CP)|.
    """
    if not V_CP_list:
        return 0

        # Combine all nodes in V_CP into a set
    V_CP_nodes = set()
    for comp in V_CP_list:
        V_CP_nodes.update(comp)

    # Count edges connecting V_CP nodes to nodes outside V_CP in a different district
    cut_edges = 0
    for node in V_CP_nodes:
        for neighbor in G.neighbors(node):
            # Edge is part of the cut if:
            # 1. Neighbor is outside V_CP.
            # 2. Neighbor belongs to a different district under the current partition.
            if neighbor not in V_CP_nodes and partition[neighbor] != partition[node]:
                cut_edges += 1

    return cut_edges


def save_partitions_as_dataframe(partitions, output_file=None):
    """
    Saves all partitions and their rewards into a Pandas DataFrame.

    Parameters:
    - partitions: List of partition objects.
    - rewards: List of corresponding reward values.
    - output_file: Optional path to save the DataFrame as a file (.csv or .pkl).

    Returns:
    - DataFrame containing partitions and rewards.
    """
    # Partitions into a DataFrame (option to add anything else)
    df = pd.DataFrame({
        'Partition': partitions
    })

    # Save the DataFrame to a file if a path is provided
    if output_file:
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.pkl'):
            df.to_pickle(output_file)
        else:
            raise ValueError("Unsupported file format. Use .csv or .pkl.")

    return df


def ensure_polygon(geometry):
    """
    Convert a geometry to a Polygon if possible
    """
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        # Optionally, return the largest polygon by area or the first one
        return max(geometry.geoms, key=lambda g: g.area)  # Largest polygon
    else:
        # Convert if possible (handles lists of coordinates)
        try:
            return Polygon(geometry)
        except Exception:
            return None  # Return None for invalid geometries


def prep_data(state_abrv):
    """
    Prepares the data for a given state abbreviation.

    Parameters:
        state_abrv (str): The state abbreviation.

    Returns:
        pandas.DataFrame: The prepared DataFrame with updated geometry and coordinates.
    """
    # Always have the working directory be the root of the project "GFN_Gerrymandering"
    data_path = f"data/{state_abrv}_raw_data.json"
    df = load_raw_data(data_path)

    # Convert the 'geometry' column to shapely Polygon objects (if needed)
    df['geometry'] = df['geometry'].apply(ensure_polygon)
    df['coordinates'] = df['geometry'].apply(lambda geom: geom.centroid)  # Compute the centroid of each polygon
    df['coordinates'] = df['coordinates'].apply(lambda point: [point.x, point.y])

    return df


def calculate_percentage_changed(partition_current, partition_original):
    """
    Calculate the percentage of precincts (nodes) that have switched districts.

    Parameters:
        partition_current (dict): Current mapping of node IDs to districts.
        partition_original (dict): Original mapping of node IDs to districts.

    Returns:
        float: Percentage of precincts that have changed districts.
    """
    total_nodes = len(partition_original)
    changed_nodes = sum(
        1 for node in partition_original
        if partition_current.get(node) != partition_original[node]
    )
    percent_changed = (changed_nodes / total_nodes) * 100
    return percent_changed


def filter_valid_samples(samples, populations, gdf, delta, compactness_threshold, compactness_constraint):
    """
    Filters samples based on population deviation and compactness constraints.
    """
    valid_samples = []
    for samp in samples:
        max_dev = max_population_deviation(samp, populations)
        if compactness_constraint:
            avg_compactness = compute_avg_compactness(samp, gdf)
            if max_dev <= delta and avg_compactness >= compactness_threshold:
                # Keep only samples that meet the population and compactness constraints
                valid_samples.append(samp)
        elif max_dev <= delta:
            # Keep only samples that meet the population constraint
            valid_samples.append(samp)
    return valid_samples


def g_func_cached(partition, populations, beta, ideal_pop):
    """
    Optimized Gibbs distribution calculation with cached ideal population.
    """
    district_pop = defaultdict(float)
    for node, dist in partition.items():
        district_pop[dist] += populations[node]

    # Compute deviation sum
    deviation_sum = sum(abs((pop / ideal_pop) - 1) for pop in district_pop.values())
    return np.exp(-beta * deviation_sum)


def accept_or_reject_proposal(current_partition, proposed_partition,
                              G, BCP_current,
                              V_CP, R, q, lambda_param,
                              g_current, g_proposed, df):
    """
    A simple Metropolis-Hastings accept/reject step. A new sample is proposed based on the previous sample,
    then the proposed sample is either added to the chain or rejected based on the acceptance probability.

    This approximation is valid under the assumption that we rarely reject samples drawn in Step 3(b) for adjacency or
    shattering issues in our method "select_nonadjacent_components".

    Assumes symmetric proposals, so the acceptance ratio is g(π')/g(π) and the ratio of the probabilities of the
    Gibbs distribution for the two partitions for the population constraint.

    Parameters:
    - current_partition: The current partition π.
    - proposed_partition: The proposed new partition π'.
    - G: The graph of the precincts.
    - BCP_current: The number of boundary components in the current partition.
    - V_CP: The set of selected boundary components (not used here, but kept for consistency).
    - R: The number of counties to change in the proposed partition.
    - q: Probability threshold for turning on edges (not used directly here).
    - lambda_param: The lambda parameter for the zero-truncated Poisson distribution.
    - g_current: The unnormalized target probability g(π).
    - g_proposed: The unnormalized target probability g(π').

    Returns:
    - A partition (dict): Either the proposed_partition if accepted, or the current_partition if rejected.
    """
    # Precomputed g(π) and g(π') - target distribution ratios

    # If both g_current and g_proposed are zero, just reject to avoid division by zero
    if g_current == 0 and g_proposed == 0:
        return False
    # If g_current == 0 but g_proposed > 0, we can set ratio to something large; effectively always accept
    if g_current == 0 and g_proposed > 0:
        return True

    # |B(CP, π)| -> Precomputed from the main loop BCP_current = |B(CP, π)|
    # Compute |B(CP, π')|
    E_on_proposed = turn_on_edges(G, proposed_partition, q)
    boundary_components_proposed = find_boundary_connected_components(G, proposed_partition, E_on_proposed)
    BCP_proposed = len(boundary_components_proposed)

    # OPT Visualization before changes (optional; can slow down execution by a lot)
    # visualize_map_with_graph_and_geometry(G, E_on_proposed, boundary_components_proposed, V_CP, df, proposed_partition)

    # Compute |C(π, V_CP)|
    C_pi_VCP = compute_swendsen_wang_cut(G, current_partition, V_CP)

    # Compute |C(π', V_CP)| by counting how V_CP is structured under π'
    C_pi_prime_VCP = compute_swendsen_wang_cut(G, proposed_partition, V_CP)

    # Compute F(|B(CP, π)|) and F(|B(CP, π')|)
    # F is the truncated Poisson pmf for the chosen R given the number of boundary components
    F_current = truncated_poisson_pmf(R, lambda_param, BCP_current)
    F_proposed = truncated_poisson_pmf(R, lambda_param, BCP_proposed)

    # Handle cases where F_proposed = 0 (no valid R under π')
    # This would make the ratio infinite, but min(1, ...) caps it at 1.
    if F_proposed == 0:
        # If it's impossible to choose R from π' scenario, ratio becomes 0.
        ratio_F = 0.0
    else:
        ratio_F = F_current / F_proposed

    # Compute (|B(CP, π)| / |B(CP, π')|)^R
    # If BCP_proposed = 0, can't form any boundary components (unlikely, but check)
    if BCP_proposed == 0:
        boundary_ratio = 0.0
    else:
        boundary_ratio = (BCP_current / BCP_proposed) ** R

    # Compute ( (1-q)^{|C(π',V_CP)|} / (1-q)^{|C(π,V_CP)|} ) = (1-q)^{C_pi_prime_VCP - C_pi_VCP}
    pq_ratio = (1 - q) ** (C_pi_prime_VCP - C_pi_VCP)

    # Compute g(π')/g(π)
    if g_current == 0:
        if g_proposed > 0:
            g_ratio = float('inf')
        else:
            g_ratio = 1.0
    else:
        g_ratio = g_proposed / g_current

    # Combine all terms:
    # α = min(1, boundary_ratio * ratio_F * pq_ratio * g_ratio)
    MH_ratio = boundary_ratio * ratio_F * pq_ratio * g_ratio
    alpha = min(1.0, MH_ratio)

    # Accept or reject
    u = random.random()
    return u <= alpha


def compute_full_partisan_bias(df, district_vector, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru",
                               step=0.01):
    """
    Compute the partisan bias measure by:
    1. Simulating uniform swings from 40% to 60% in statewide Dem share.
    2. Constructing the seats-votes curve f(x).
    3. Comparing f(x) to a symmetric reference f*(x) and integrating the difference.

    Args:
        df (pd.DataFrame): DataFrame with vote data per unit.
        district_vector (list or pd.Series): District assignment for each unit.
        dem_vote_col (str): Column name for Democratic votes.
        rep_vote_col (str): Column name for Republican votes.
        step (float): Increment for simulation steps (e.g., 0.01 for 1% increments).

    Returns:
        float: The computed partisan bias measure.
    """
    df = df.copy()
    df['district'] = district_vector
    district_votes = df.groupby('district')[[dem_vote_col, rep_vote_col]].sum().reset_index()
    district_votes['total_votes'] = district_votes[dem_vote_col] + district_votes[rep_vote_col]

    # Statewide totals
    statewide_dem = district_votes[dem_vote_col].sum()
    statewide_rep = district_votes[rep_vote_col].sum()
    statewide_total = statewide_dem + statewide_rep
    original_dem_share = statewide_dem / statewide_total

    # Range of x values from 40% to 60%
    lower_bound = 0.4
    upper_bound = 0.6
    x_values = np.arange(lower_bound, upper_bound + step, step)

    def compute_seat_share_for_x(x):
        delta = x - original_dem_share
        district_votes['dem_share'] = district_votes[dem_vote_col] / district_votes['total_votes']
        district_votes['dem_share_adj'] = district_votes['dem_share'] + delta
        district_votes['dem_share_adj'] = district_votes['dem_share_adj'].clip(0, 1)

        district_votes['dem_votes_adj'] = district_votes['dem_share_adj'] * district_votes['total_votes']
        district_votes['rep_votes_adj'] = (1 - district_votes['dem_share_adj']) * district_votes['total_votes']

        dem_wins = (district_votes['dem_votes_adj'] > district_votes['rep_votes_adj']).sum()
        total_districts = len(district_votes)
        return dem_wins / total_districts

    # Compute f(x)
    f_values = np.array([compute_seat_share_for_x(x) for x in x_values])

    # Symmetric baseline f*(x): linear from f*(0.4)=0 to f*(0.6)=1
    f_star_values = (x_values - 0.4) / 0.2

    # Integrate difference [f(x)-f*(x)]
    differences = f_values - f_star_values
    integral = np.trapz(differences, x_values)

    eta = 0.1
    partisan_bias = (1 / eta) * integral

    return partisan_bias
