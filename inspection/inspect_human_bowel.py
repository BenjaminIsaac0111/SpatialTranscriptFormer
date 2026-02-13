import pandas as pd
import os

potential_paths = [
    "hest_data/HEST_v1_3_0.csv",
    "../hest_data/HEST_v1_3_0.csv",
    "HEST_v1_3_0.csv",
    "../HEST_v1_3_0.csv",
    r"A:\hest_data\HEST_v1_3_0.csv"
]

csv_path = None
for p in potential_paths:
    if os.path.exists(p):
        csv_path = p
        break

if not csv_path:
    print(f"Error: Could not find HEST_v1_3_0.csv in any of {potential_paths}")
else:
    print(f"Reading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    bowel_df = df[df['organ'].str.lower() == 'bowel']
    print("Bowel samples species:")
    print(bowel_df['species'].value_counts())

    print("\nSample IDs for Human Bowel:")
    human_bowel = bowel_df[bowel_df['species'] == 'Homo sapiens']
    print(human_bowel['id'].head(10).tolist())
