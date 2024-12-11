import json
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape, Point, box
from math import atan2, radians, sin, cos, pi

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

    Example usage:
    ```
        metrics = {
            "total": [("vap", "Voting Age Population")],
            "mean": [],
            "ratio": [[("pre_20_dem_bid", "Biden"), ("pre_20_rep_tru", "Trump")]]
        }
        visualize_map_with_geometry(df, geometry_col="geometry", district_id_col="cd_2020", state="Iowa", metrics=metrics)
    ```
    """
    # Basic setup
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    
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

            if "total" in metrics:
                if metrics["total"]:
                    for col, var_name in metrics["total"]:
                        total = district_data[col].sum()
                        metric_texts.append(f"Total {var_name}: {total:,}")
                        metric_texts.append("━━━━━━━━━━━")

            if "mean" in metrics:
                if metrics["mean"]:
                    for col, var_name in metrics["mean"]:
                        mean = district_data[col].mean()
                        metric_texts.append(f"Mean {var_name}: {mean:.2f}")
                        metric_texts.append("━━━━━━━━━━━")

            if "ratio" in metrics:
                if metrics["ratio"]:
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
        
        # Start from outside the boundary
        min_distance = 0.1 * (maxx - minx)
        max_distance = 1 * (maxx - minx)
        distance_increment = 0.01 * (maxx - minx)
        angle_range = pi/2
        
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

    plt.show()