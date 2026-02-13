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
    try:
        print(f"Reading metadata from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(df[['id', 'organ', 'species', 'disease_state']].head(20))
        print("\nSpecies counts:")
        print(df['species'].value_counts())
        print("\nOrgan counts:")
        print(df['organ'].value_counts())
    except Exception as e:
        print(e)
