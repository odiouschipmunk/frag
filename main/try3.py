import pandas as pd
import json

# Define all attributes
attributes = [
    "Base notes", "Middle notes", "Top notes", "Notes", "Name", "Perfumer", 
    "Description", "Gender", "Concepts", "URL", "All reviews", 
    "Positive reviews", "Negative reviews", "Image URL"
]

# Read CSV file with specified encoding
csv_df = pd.read_csv(r'C:\Users\default.DESKTOP-7FKFEEG\frag\main\data\final_perfume_data.csv', encoding='ISO-8859-1')


# Ensure all DataFrames have the same columns
def normalize_df(df, attributes):
    for attr in attributes:
        if attr not in df.columns:
            df[attr] = "N/A"
    return df[attributes]

csv_df = normalize_df(csv_df, attributes)

# Merge DataFrames
merged_df = pd.concat([csv_df], ignore_index=True)

# Save to JSON
merged_df.to_json('final.json', orient='records', lines=True)