import json
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape, Point, box
from math import atan2, radians, sin, cos, pi
import numpy as np
import ast
import ipywidgets as widgets
import seaborn as sns
from IPython.display import display, clear_output

################
# COMPUTATIONS #
################

def load_raw_data(path):
    # Read JSON raw file
    with open(path, 'r') as file:
        data = json.load(file)

    # Flatten the JSON structure into a DataFrame
    df = pd.json_normalize(data)

    # Check if required columns for geometry exist
    if "geometry.type" in df.columns and "geometry.coordinates" in df.columns:
        # Add geometry column
        df['geometry'] = df.apply(
            lambda row: {"type": row["geometry.type"], "coordinates": row["geometry.coordinates"]},
            axis=1
        )
        df['geometry'] = df['geometry'].apply(shape)

        # Create and return a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        return gdf

    # Return Pandas DataFrame if geometry columns are not present
    return df


def compute_efficiency_gap(df, district_vector, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru"):
    """
    Compute the efficiency gap given the actual election results.

    Args:
        df (pd.DataFrame): DataFrame with vote data.
        district_vector (list or pd.Series): District assignments.
        dem_vote_col (str): Column name for Democratic votes.
        rep_vote_col (str): Column name for Republican votes.

    Returns:
        float: The efficiency gap.
    """
    df = df.copy()
    df['district'] = district_vector
    district_votes = df.groupby('district')[[dem_vote_col, rep_vote_col]].sum().reset_index()
    district_votes['total_votes'] = district_votes[dem_vote_col] + district_votes[rep_vote_col]

    statewide_total = district_votes['total_votes'].sum()

    district_votes['eff_gap_winner'] = district_votes.apply(
        lambda row: 'Democrat' if row[dem_vote_col] > row[rep_vote_col] else 'Republican',
        axis=1
    )

    # threshold = floor(total_votes/2) + 1
    district_votes['threshold'] = (district_votes['total_votes'] // 2) + 1

    # Calculate wasted votes
    district_votes['dem_wasted'] = district_votes.apply(
        lambda row: (row[dem_vote_col] - row['threshold']) if row['eff_gap_winner'] == 'Democrat' else row[dem_vote_col],
        axis=1
    )

    district_votes['rep_wasted'] = district_votes.apply(
        lambda row: (row[rep_vote_col] - row['threshold']) if row['eff_gap_winner'] == 'Republican' else row[rep_vote_col],
        axis=1
    )

    total_dem_wasted = district_votes['dem_wasted'].sum()
    total_rep_wasted = district_votes['rep_wasted'].sum()

    efficiency_gap = (total_dem_wasted - total_rep_wasted) / statewide_total
    return efficiency_gap


def compute_partisan_bias(df, district_vector, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru", v=0.5):
    """
    Compute the partisan bias as described by the formula:
    Partisan Bias = (Seats_D(v) - [1 - Seats_D(1-v)]) / 2

    Args:
        df (pd.DataFrame): DataFrame with vote data per unit.
        district_vector (list or pd.Series): District assignment for each unit.
        dem_vote_col (str): Column name for Democratic votes.
        rep_vote_col (str): Column name for Republican votes.
        v (float): The targeted Democratic share of the vote, default 0.5.

    Returns:
        float: The computed partisan bias measure.
    """
    df = df.copy()
    df['district'] = district_vector
    district_votes = df.groupby('district')[[dem_vote_col, rep_vote_col]].sum().reset_index()
    district_votes['total_votes'] = district_votes[dem_vote_col] + district_votes[rep_vote_col]

    # Original statewide totals
    statewide_dem = district_votes[dem_vote_col].sum()
    statewide_rep = district_votes[rep_vote_col].sum()
    statewide_total = statewide_dem + statewide_rep
    original_dem_share = statewide_dem / statewide_total

    def seats_d_given_share(target_share):
        # Adjust each district's dem share by delta so that total dem share = target_share
        delta = target_share - original_dem_share

        # Current district-level dem share
        district_votes['dem_share'] = district_votes[dem_vote_col] / district_votes['total_votes']
        # Adjusted dem share, clipped between 0 and 1
        district_votes['dem_share_adj'] = (district_votes['dem_share'] + delta).clip(0, 1)

        # Adjusted votes
        district_votes['dem_votes_adj'] = district_votes['dem_share_adj'] * district_votes['total_votes']
        district_votes['rep_votes_adj'] = (1 - district_votes['dem_share_adj']) * district_votes['total_votes']

        # Count seats won by Democrats
        dem_wins = (district_votes['dem_votes_adj'] > district_votes['rep_votes_adj']).sum()
        total_districts = len(district_votes)
        return dem_wins / total_districts

    seats_d_at_v = seats_d_given_share(v)
    seats_d_at_1minusv = seats_d_given_share(1 - v)

    partisan_bias = (seats_d_at_v - (1 - seats_d_at_1minusv)) / 2
    return -partisan_bias # -1 to return republican bias like in the paper


def compute_compactness(df, district_vector):
    """
    Compute the mean Polsby-Popper compactness score for districts defined by a district vector.

    Args:
        df (gpd.GeoDataFrame): DataFrame with county geometries in the 'geometry' column.
        district_vector (list or array): Vector of district identifiers (same length as df).

    Returns:
        float: The mean Polsby-Popper compactness score across all districts.
    """
    # Assign the district vector to a new column in the DataFrame
    df = df.copy()
    df['district'] = district_vector

    # Group by the new 'district' column and create a unified geometry for each district
    grouped = df.groupby('district')['geometry'].apply(lambda x: x.union_all()).reset_index()

    # List to store compactness scores
    compactness_scores = []

    for _, row in grouped.iterrows():
        geom = row['geometry']

        # Compute the Polsby-Popper compactness score
        area = geom.area  # District area
        perimeter = geom.length  # District perimeter
        score = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Add the score to the list
        compactness_scores.append(score)

    # Return the mean compactness score across all districts
    return np.mean(compactness_scores) if compactness_scores else 0

#################
# VISUALIZATION #
#################

def plot_district_swaps(df, new_districts, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru"):
    """
    Creates an interactive plot with dropdown selection showing KDE plot and district map side by side.
    
    Parameters:
    df: GeoDataFrame with district data
    new_districts: array of new district assignments
    dem_vote_col: column name for Democratic votes
    rep_vote_col: column name for Republican votes
    """
    def create_plot(district_id):
        # Clear previous output and recreate dropdown
        clear_output(wait=True)
        display(district_dropdown)
        
        # Extract data for the current district
        current_district = df[df['cd_2010'] == district_id]
        initial_indexes = list(current_district.index)
        new_indexes = list(np.where(new_districts == district_id)[0])
        
        # Identify unchanged, swapped-in, and swapped-out counties
        unchanged = set(initial_indexes).intersection(new_indexes)
        swap_out = set(initial_indexes) - unchanged
        swap_in = set(new_indexes) - unchanged
        
        # Create subsets for vote share calculations
        unchanged_df = df.loc[list(unchanged)]
        swap_out_df = df.loc[list(swap_out)]
        swap_in_df = df.loc[list(swap_in)]
        
        # Calculate Democratic vote ratios
        def calc_dem_ratio(subset):
            return subset[dem_vote_col] / (subset[dem_vote_col] + subset[rep_vote_col])
        
        unchanged_votes = calc_dem_ratio(unchanged_df)
        swap_out_votes = calc_dem_ratio(swap_out_df)
        swap_in_votes = calc_dem_ratio(swap_in_df)
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # KDE Plot with handling for single counties
        if len(unchanged_votes) > 1:
            sns.kdeplot(unchanged_votes, label='Not Swapped', color='black', ax=axes[0])
        elif len(unchanged_votes) == 1:
            axes[0].axvline(unchanged_votes.iloc[0], color='black', label='Not Swapped')
            
        if len(swap_out_votes) > 1:
            sns.kdeplot(swap_out_votes, label='Swapped Out', color='blue', linestyle='--', ax=axes[0])
        elif len(swap_out_votes) == 1:
            axes[0].axvline(swap_out_votes.iloc[0], color='blue', linestyle='--', label='Swapped Out')
            
        if len(swap_in_votes) > 1:
            sns.kdeplot(swap_in_votes, label='Swapped In', color='red', linestyle='-.', ax=axes[0])
        elif len(swap_in_votes) == 1:
            axes[0].axvline(swap_in_votes.iloc[0], color='red', linestyle='-.', label='Swapped In')
        
        axes[0].set_xlabel('Vote Share for Democratic Congressional Candidate')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution of Democratic Voteshare for District {district_id}')
        axes[0].legend()
        
        # District Map
        current_district.plot(ax=axes[1], color='white', edgecolor='black', linewidth=0.5)
        if not swap_out_df.empty:
            swap_out_df.plot(ax=axes[1], color='blue', label='Swapped Out')
        if not swap_in_df.empty:
            swap_in_df.plot(ax=axes[1], color='red', label='Swapped In')
        
        handles = [
            plt.Line2D([0], [0], color='blue', lw=4, label='Swapped Out'),
            plt.Line2D([0], [0], color='red', lw=4, label='Swapped In')
        ]
        axes[1].legend(handles=handles)
        axes[1].set_title(f'Congressional District {district_id}')
        
        plt.tight_layout()
        plt.show()
    
    # Get unique districts and create dropdown
    districts = sorted(df['cd_2020'].unique())
    district_dropdown = widgets.Dropdown(
        options=districts,
        value=districts[0],
        description='District:',
        style={'description_width': 'initial'}
    )
    
    # Connect the dropdown to the plot update function
    district_dropdown.observe(lambda change: create_plot(change.new), names='value')
    
    # Show initial plot
    create_plot(districts[0])


def partisan_bias_vs_presincts_changed(initial_districts, proposed_partitions):
    """
    Generate a plot of partisan bias vs. percentage of precincts switched from the original district.

    Args:
        initial_districts (array-like): Initial district assignments.
        proposed_partitions (pd.DataFrame): DataFrame with proposed partitions and rewards.

    Returns:
        None: Displays a plot.
    """
    initial_districts = np.array(initial_districts)
    points = []

    for row in proposed_partitions.itertuples():
        partition = row.Partition
        districts = np.array(list(ast.literal_eval(partition).values()))
        diff = initial_districts != districts
        ptg = np.sum(diff) / len(districts)  # Percentage of switched precincts
        points.append((ptg, compute_partisan_bias(df, district_vector=districts)))

    # Split points into x and y for plotting
    x, y = zip(*points)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="black", s=10, alpha=0.7, label="Simulated Plans")
    plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Unbiased Plan (0 Bias)")
    plt.axvline(0.03, color="blue", linestyle="--", linewidth=1, label="Minimal Bias Threshold (3%)")
    
    # Customize aesthetics
    plt.title("Partisan Bias of Simulated Plans", fontsize=14)
    plt.xlabel("% of Precincts Switched From Original District", fontsize=12)
    plt.ylabel("Partisan Bias towards Democrats", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Show the plot
    plt.show()

# First, add a new function to calculate required margins after box placement
def calculate_required_margins(placed_boxes, minx, miny, maxx, maxy):
    """Calculate minimum required margins for each side based on box positions"""
    if not placed_boxes:
        default = 0.2 * (maxx - minx)
        return default, default, default, default
    
    left_margin = 0
    right_margin = 0
    top_margin = 0
    bottom_margin = 0
    
    for (box_minx, box_miny, box_maxx, box_maxy) in placed_boxes:
        if box_minx < minx:
            left_margin = max(left_margin, minx - box_minx)
        if box_maxx > maxx:
            right_margin = max(right_margin, box_maxx - maxx)
        if box_miny < miny:
            bottom_margin = max(bottom_margin, miny - box_miny)
        if box_maxy > maxy:
            top_margin = max(top_margin, box_maxy - maxy)
    
    # Add some padding to each margin independently
    padding = 0.05 * (maxx - minx)
    left_margin += padding
    right_margin += padding
    top_margin += padding
    bottom_margin += padding
    
    return left_margin, right_margin, top_margin, bottom_margin


def visualize_map_with_geometry(df, geometry_col, district_id_col, state, metrics=None):
    """
    Visualizes a map where precincts are colored based on the district ID,
    with non-overlapping metric boxes placed close to their districts.
    """

    # Check if district_id_col is a column name or an array
    if isinstance(district_id_col, str):
        gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    else:
        # district_id_col is assumed to be an array-like object
        gdf = gpd.GeoDataFrame(df.copy(), geometry=df[geometry_col])
        gdf['district_id'] = district_id_col
        district_id_col = 'district_id'

    unique_districts = gdf[district_id_col].unique()
    cmap = plt.cm.get_cmap("tab20", len(unique_districts))
    district_colors = {district: cmap(i) for i, district in enumerate(unique_districts)}
    gdf['color'] = gdf[district_id_col].map(district_colors)

    # Calculate district geometries early
    district_geoms = {d: gdf[gdf[district_id_col] == d].geometry.union_all()
                      for d in unique_districts}
    centroids = {d: district_geoms[d].centroid.coords[0] for d in unique_districts}

    # Calculate metrics
    aggregated_metrics = {}
    if metrics:
        grouped = gdf.groupby(district_id_col)
        for district in unique_districts:
            district_data = grouped.get_group(district)
            metric_texts = []

            if "total" in metrics and metrics["total"]:
                for col, var_name in metrics["total"]:
                    total = district_data[col].sum()
                    metric_texts.append(f"Total {var_name}: {total:,}")
                    metric_texts.append("━━━━━━━━━━━")

            if "mean" in metrics and metrics["mean"]:
                for col, var_name in metrics["mean"]:
                    mean = district_data[col].mean()
                    metric_texts.append(f"Mean {var_name}: {mean:.2f}")
                    metric_texts.append("━━━━━━━━━━━")

            if "ratio" in metrics and metrics["ratio"]:
                for group in metrics["ratio"]:
                    cols = [x[0] for x in group]  # Get column names
                    group_total = district_data[cols].sum(axis=0).sum()
                    for col, var_name in group:
                        ratio = (district_data[col].sum() / group_total * 100) if group_total != 0 else 0
                        metric_texts.append(f"{var_name}: {ratio:.1f}%")
                    metric_texts.append("━━━━━━━━━━━")

            aggregated_metrics[district] = "\n".join(metric_texts)

    # Create figure
    fig, ax = plt.subplots(figsize=(25, 20))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black')
    plt.title(f"District Visualization for {state} State")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Start with initial margins
    minx, miny, maxx, maxy = gdf.total_bounds
    x_margin = 0.3 * (maxx - minx)  # Initial margins
    y_margin = 0.3 * (maxy - miny)

    # Create state boundary
    state_boundary = gdf.union_all().boundary
    state_box = box(minx, miny, maxx, maxy)

    def nearest_edge_point(geom):
        if not isinstance(geom, Point):
            geom = geom.centroid
        return state_boundary.interpolate(state_boundary.project(geom))

    def is_enclaved(district_geom):
        return not district_geom.intersects(state_boundary)

    placed_boxes = []

    def does_overlap(x_min, y_min, x_max, y_max, padding=0.04):
        pad_x = (maxx - minx) * padding
        pad_y = (maxy - miny) * padding

        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        # Check map bounds
        if (x_min < minx - x_margin or x_max > maxx + x_margin or
                y_min < miny - y_margin or y_max > maxy + y_margin):
            return True

        # Check against existing boxes with padding
        for (Xmin, Ymin, Xmax, Ymax) in placed_boxes:
            if not (x_max < Xmin - pad_x or x_min > Xmax + pad_x or
                    y_max < Ymin - pad_y or y_min > Ymax + pad_y):
                return True
        return False

    def estimate_box_size(text):
        lines = text.count('\n') + 1
        max_line_len = max(len(line) for line in text.split('\n')) if text else 10
        map_width = maxx - minx
        map_height = maxy - miny

        char_width_factor = 0.003 * map_width
        line_height_factor = 0.002 * map_height

        box_width = max_line_len * char_width_factor * 2.5
        box_height = lines * line_height_factor * 4.0

        return box_width, box_height

    # Modify find_placement function to ensure boxes are outside state boundary
    def find_placement(centroid, box_width, box_height, base_angle):
        cx, cy = centroid

        # Get the state polygon (not just boundary)
        state_polygon = gdf.union_all()

        # First find the nearest point on the state boundary
        nearest = nearest_edge_point(Point(cx, cy))
        start_x, start_y = nearest.x, nearest.y

        # Calculate direction pointing outward from the state
        outward_angle = atan2(start_y - cy, start_x - cx)

        min_distance = 0.1 * (maxx - minx)
        max_distance = 1 * (maxx - minx)
        distance_increment = 0.01 * (maxx - minx)
        angle_range = pi / 2

        search_distance = min_distance
        while search_distance < max_distance:
            for angle_offset in range(-12, 13):
                angle = outward_angle + (angle_offset * angle_range / 12)

                x_try = start_x + search_distance * cos(angle)
                y_try = start_y + search_distance * sin(angle)

                x_min = x_try - box_width / 2
                x_max = x_try + box_width / 2
                y_min = y_try - box_height / 2
                y_max = y_try + box_height / 2

                # Create test box
                test_box = box(x_min, y_min, x_max, y_max)

                # Strict check: box must not intersect state polygon
                if (not test_box.intersects(state_polygon) and
                        not does_overlap(x_min, y_min, x_max, y_max)):
                    return x_try, y_try, x_min, y_min, x_max, y_max

            search_distance += distance_increment

        return None

    # Sort districts by size and position
    district_sizes = {d: district_geoms[d].area for d in unique_districts}
    sorted_districts = sorted(unique_districts, key=lambda d: district_sizes[d], reverse=True)

    for d in sorted_districts:
        text = aggregated_metrics.get(d, "")
        box_color = district_colors[d]
        cx, cy = centroids[d]
        w, h = estimate_box_size(text)

        district_geom = district_geoms[d]
        if is_enclaved(district_geom):
            nearest_point = nearest_edge_point(district_geom)
            ref_x, ref_y = nearest_point.x, nearest_point.y
            x_arrow, y_arrow = cx, cy
        else:
            nearest_point = nearest_edge_point(district_geom)
            ref_x, ref_y = nearest_point.x, nearest_point.y
            x_arrow, y_arrow = ref_x, ref_y

        base_angle = atan2(ref_y - cy, ref_x - cx)

        placement = find_placement((cx, cy), w, h, base_angle)

        if placement:
            x_try, y_try, x_min, y_min, x_max, y_max = placement

            ax.annotate(
                text,
                xy=(x_arrow, y_arrow),
                xycoords="data",
                xytext=(x_try, y_try),
                textcoords="data",
                ha="center",
                va="center",
                bbox=dict(facecolor=box_color, alpha=0.9, boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color="black"),
            )
            placed_boxes.append((x_min, y_min, x_max, y_max))

    # After placing all boxes, calculate required margins for each side
    left_margin, right_margin, top_margin, bottom_margin = calculate_required_margins(
        placed_boxes, minx, miny, maxx, maxy
    )

    # Update plot limits with calculated margins
    ax.set_xlim(minx - left_margin, maxx + right_margin)
    ax.set_ylim(miny - bottom_margin, maxy + top_margin)

    # Remove white borders and axis to see the map better
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.show()