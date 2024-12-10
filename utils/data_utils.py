import json
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

def load_raw_data(path):

    # Read JSON raw file
    with open(path, 'r') as file:
        data = json.load(file)

    # Flatten the JSON structure into a DataFrame
    df = pd.json_normalize(data)
    return df


def visualize_map_with_geometry(df, geometry_col, district_id_col, state):
    """
    Visualizes a map where precincts are colored based on the district ID,
    directly using the geometry column.

    Args:
        df (pd.DataFrame): The dataframe containing the map data.
        geometry_col (str): The column name for geometry (e.g., 'geometry').
        district_id_col (str): The column name for district IDs.
    """
    # Ensure the geometry column is set correctly
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col])
    
    # Generate a colormap for districts
    unique_districts = gdf[district_id_col].unique()
    cmap = plt.cm.get_cmap("tab20", len(unique_districts))
    district_colors = {district: cmap(i) for i, district in enumerate(unique_districts)}
    
    # Map the colors to the GeoDataFrame
    gdf['color'] = gdf[district_id_col].map(district_colors)
    
    # Plot the map
    plt.figure(figsize=(12, 10))
    gdf.plot(
        ax=plt.gca(),
        color=gdf['color'],
        edgecolor='black'
    )
    plt.title(f"District Visualization for {state} State")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()