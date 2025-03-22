# # data_processing.py
# import os
# import pandas as pd
# import numpy as np
# import openpyxl

# def load_metadata(metadata_file="metadata.xlsx"):
#     """
#     Loads metadata from the Excel file.
#     Expects the file to have columns: Filename, Location, Variable, Unit.
#     """
#     try:
#         metadata = pd.read_excel(metadata_file)
#     except Exception as e:
#         raise FileNotFoundError(f"Error loading metadata file '{metadata_file}': {e}")
#     return metadata

# def load_csv(file_path, col_name):
#     """
#     Loads a CSV file and renames its measurement column to a standardized name.
#     Assumes the CSV has at least two columns: DateTime and the measurement.
#     """
#     df = pd.read_csv(file_path)
#     # Ensure DateTime column is parsed as datetime.
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     # Assume the measurement column is the second column
#     measurement = df.columns[1]
#     df = df.rename(columns={measurement: col_name})
#     # Replace missing values (-9999) with NaN
#     df[col_name] = df[col_name].replace(-9999, np.nan)
#     return df

# def merge_data_for_location(location, data_dir="data", metadata_file="metadata.xlsx"):
#     """
#     For a given location, loads the corresponding CSV files based on metadata,
#     and merges them on the DateTime column.
    
#     Returns a merged DataFrame for that location.
#     """
#     metadata = load_metadata(metadata_file)
#     # Filter metadata for the given location
#     loc_meta = metadata[metadata["Location"] == location]
#     if loc_meta.empty:
#         raise ValueError(f"No metadata found for location '{location}'")
    
#     dfs = []
#     for idx, row in loc_meta.iterrows():
#         filename = row["Filename"]
#         variable = row["Variable"]
#         file_path = os.path.join(data_dir, filename)
#         try:
#             df = load_csv(file_path, variable)
#             dfs.append(df)
#         except Exception as e:
#             print(f"Error loading file '{filename}': {e}")
    
#     if not dfs:
#         raise ValueError("No CSV files loaded for this location.")
    
#     # Merge all DataFrames on DateTime using inner join
#     merged_df = dfs[0]
#     for df in dfs[1:]:
#         merged_df = pd.merge(merged_df, df, on="DateTime", how="inner")
    
#     # Set DateTime as index before interpolation to avoid attempting to interpolate non-numeric columns.
#     merged_df = merged_df.set_index("DateTime").interpolate(method='time').reset_index()
#     return merged_df

# if __name__ == "__main__":
#     # Example: Process and print merged data for the "MountainTower" location.
#     try:
#         mountain_tower_data = merge_data_for_location("MountainTower")
#         print(mountain_tower_data.head())
#     except Exception as e:
#         print("Error:", e)

# data_processing.py
import os
import pandas as pd
import numpy as np

def get_location_from_filename(file_path):
    """
    Extracts the location from the CSV file name.
    Rule:
      - If the file name starts with 'RF' (case-insensitive), return 'RainForest'.
      - Otherwise, return the first word (before the first underscore).
    """
    base = os.path.basename(file_path)
    parts = base.split("_")
    if parts[0].upper().startswith("RF"):
        return "RainForest"
    else:
        return parts[0]

def get_variable_from_filename(file_path):
    """
    Derives a variable name from the file name.
    For example, for "RF_MountainTower_rad_at10m_FEB-2025.csv",
    it joins parts [1:-1] to produce "MountainTower_rad_at10m".
    If not enough parts exist, returns "Measurement".
    """
    base = os.path.basename(file_path)
    parts = base.split("_")
    if len(parts) > 2:
        variable = "_".join(parts[1:-1])
    else:
        variable = "Measurement"
    return variable

def load_csv_with_location(file_path):
    """
    Loads a CSV file, ensuring the DateTime column is parsed,
    extracts the location and variable from the file name,
    adds a 'Location' column, renames the first measurement column
    to a unique name, and drops any additional columns.
    """
    df = pd.read_csv(file_path)
    
    # Rename first column to DateTime if needed.
    if "DateTime" not in df.columns:
        df.rename(columns={df.columns[0]: "DateTime"}, inplace=True)
    
    # Parse DateTime column.
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    
    # Extract location and variable from file name.
    location = get_location_from_filename(file_path)
    variable = get_variable_from_filename(file_path)
    
    # Create a unique measurement column name.
    unique_col = f"{location}_{variable}"
    
    # Add the Location column.
    df["Location"] = location
    
    # Identify all measurement columns (all except DateTime and Location).
    data_cols = [col for col in df.columns if col not in ["DateTime", "Location"]]
    if data_cols:
        # Choose the first measurement column and rename it.
        measurement = data_cols[0]
        df.rename(columns={measurement: unique_col}, inplace=True)
        # Drop any extra measurement columns.
        extra_cols = [col for col in data_cols if col != measurement]
        if extra_cols:
            df = df.drop(columns=extra_cols)
    else:
        # If no measurement column is found, create an empty column.
        df[unique_col] = np.nan
    
    # Replace missing numeric values (-9999) with NaN.
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].replace(-9999, np.nan)
    
    # Finally, keep only DateTime, Location, and the unique measurement column.
    df = df[["DateTime", "Location", unique_col]]
    
    return df

def load_all_csv(data_dir="data"):
    """
    Loads all CSV files from the given directory using the above helper.
    Returns a list of DataFrames.
    """
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = []
    for file in csv_files:
        try:
            df = load_csv_with_location(file)
            print(f"Loaded {file}: columns: {df.columns.tolist()}")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return dfs

def merge_by_location(dfs):
    """
    Groups the list of DataFrames by their Location value,
    then merges (outer join) all DataFrames in each group on ["DateTime", "Location"].
    Finally, sorts each merged DataFrame by DateTime and interpolates numeric columns.
    Returns a dictionary mapping location names to merged DataFrames.
    """
    groups = {}
    for df in dfs:
        loc = df["Location"].iloc[0]
        groups.setdefault(loc, []).append(df)
    
    merged_by_location = {}
    for loc, group in groups.items():
        merged_df = group[0]
        for df in group[1:]:
            merged_df = pd.merge(merged_df, df, on=["DateTime", "Location"], how="outer")
        merged_df = merged_df.sort_values("DateTime")
        merged_df = merged_df.set_index("DateTime").interpolate(method="time").reset_index()
        merged_by_location[loc] = merged_df
    return merged_by_location

if __name__ == "__main__":
    # Load all CSV files from the 'data' folder.
    dfs = load_all_csv(data_dir="data")
    if not dfs:
        raise ValueError("No CSV files loaded. Check the data directory.")
    
    # Merge DataFrames for each location.
    merged_by_location = merge_by_location(dfs)
    
    # Concatenate all location-specific DataFrames into one.
    merged_all = pd.concat(merged_by_location.values(), ignore_index=True)
    
    # Save the merged data to a CSV file.
    merged_all.to_csv("merged_data.csv", index=False)
    print("Merged data saved to merged_data.csv")
    
    # Optional: print a summary per location.
    for loc, df in merged_by_location.items():
        print(f"Location: {loc}, shape: {df.shape}")
