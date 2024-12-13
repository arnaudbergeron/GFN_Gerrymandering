import random
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


def run_algorithm_1(df, q, num_iterations):
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Need this for reward calculations
    populations = df.set_index('node_id')['pop'].to_dict()
    votes_dem = df.set_index('node_id')['pre_20_dem_bid'].to_dict()
    votes_rep = df.set_index('node_id')['pre_20_rep_tru'].to_dict()
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

    current_partition = initial_partition.copy()

    # Update the DataFrame with the new partition to visualize
    district = np.array([current_partition[node] for node in range(len(df))])
    df['district'] = district

    samples = []

    for t in range(num_iterations):
        E_on = turn_on_edges(G, current_partition, q)
        boundary_components = find_boundary_connected_components(G, current_partition, E_on)

        V_CP = select_nonadjacent_components(boundary_components, G, current_partition)
        proposed_partition = propose_swaps(current_partition, V_CP, G)

        visualize_map_with_graph_and_geometry(G, E_on, boundary_components, df, 'district')

        # Update the DataFrame with the new partition to visualize
        district = np.array([proposed_partition[node] for node in range(len(df))])
        df['district'] = district

        # Evaluate rewards
        current_reward = combined_reward(current_partition, populations, votes_dem, votes_rep)
        proposed_reward = combined_reward(proposed_partition, populations, votes_dem, votes_rep)

        current_partition = proposed_partition.copy()

        # Simple acceptance criterion: accept if proposed is better or equal
        if proposed_reward >= current_reward:
            current_partition = proposed_partition

        samples.append(current_partition.copy())

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


def select_nonadjacent_components(boundary_components, G, partition):
    if len(boundary_components) == 0:
        return []
    chosen_component = random.choice(boundary_components)
    return [chosen_component]


def propose_swaps(current_partition, V_CP, G):
    proposed_partition = current_partition.copy()

    for component in V_CP:
        current_district_set = {proposed_partition[node] for node in component}
        if len(current_district_set) != 1:
            continue
        current_district = current_district_set.pop()
        neighbor_districts = find_neighboring_districts(component, proposed_partition, G)
        if len(neighbor_districts) > 0:
            new_district = random.choice(neighbor_districts)
            for node in component:
                proposed_partition[node] = new_district

    return proposed_partition


def find_neighboring_districts(component, partition, G):
    component_district = {partition[n] for n in component}.pop()
    adjacent_districts = set()
    for node in component:
        for neigh in G.neighbors(node):
            if neigh not in component and partition[neigh] != component_district:
                adjacent_districts.add(partition[neigh])
    return list(adjacent_districts)


# def build_subgraph(E_on, nodes_to_include):
#     """
#     Construct a subgraph containing only the specified nodes and edges.
#     """
#     subgraph = {n: [] for n in nodes_to_include}
#     for (u, v) in E_on:
#         if u in nodes_to_include and v in nodes_to_include:
#             subgraph[u].append(v)
#             subgraph[v].append(u)
#     return subgraph
#
#
# def get_connected_components(adj_list):
#     visited = set()
#     components = []
#     for node in adj_list:
#         if node not in visited:
#             comp = set()
#             stack = [node]
#             while stack:
#                 curr = stack.pop()
#                 if curr not in visited:
#                     visited.add(curr)
#                     comp.add(curr)
#                     for neigh in adj_list[curr]:
#                         if neigh not in visited:
#                             stack.append(neigh)
#             components.append(comp)
#     return components

def main():
    random.seed(42)
    data_path = "data/IA_raw_data.json"
    df = load_raw_data(data_path)

    # largest amount of adjacent nodes:
    # max(list(map(len, graph.values())))

    q = 0.07  # 0.05 or 0.04 (for PA) from the paper
    num_iterations = 10000
    # Convert the 'geometry' column to shapely Polygon objects (if needed)
    df['geometry'] = df['geometry'].apply(lambda x: Polygon(x) if not isinstance(x, Polygon) else x)
    # Compute the centroid
    df['centroid'] = df['geometry'].apply(lambda geom: geom.centroid)
    df['coordinates'] = df['centroid'].apply(lambda point: [point.x, point.y])

    samples = run_algorithm_1(df, q, num_iterations)

    df = df.reset_index(drop=True)
    df['node_id'] = df.index

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
