import json
import pandas as pd

def load_raw_data(path):

    # Read JSON raw file
    with open(path, 'r') as file:
        data = json.load(file)

    # Flatten the JSON structure into a DataFrame
    df = pd.json_normalize(data)
    return df