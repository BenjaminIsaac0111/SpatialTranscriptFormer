import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from typing import List, Tuple

import argparse

def split_hest_patients(metadata_path: str, val_ratio: float = 0.2, test_ratio: float = 0.0, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits HEST samples into train/val/test based on Patient ID to prevent data leakage.
    Samples with missing patient IDs are treated as unique patients (safe fallback).
    """
    df = pd.read_csv(metadata_path)
    df['patient_filled'] = df['patient'].fillna(df['id'])
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=df['patient_filled']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    val_ids = []
    test_ids = []
    
    if test_ratio > 0:
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        test_relative_ratio = 1.0 - val_relative_ratio
        
        if len(temp_df['patient_filled'].unique()) > 1:
             splitter_2 = GroupShuffleSplit(n_splits=1, test_size=test_relative_ratio, random_state=seed)
             val_idx, test_idx = next(splitter_2.split(temp_df, groups=temp_df['patient_filled']))
             val_ids = temp_df.iloc[val_idx]['id'].tolist()
             test_ids = temp_df.iloc[test_idx]['id'].tolist()
        else:
            val_ids = temp_df['id'].tolist()
    else:
        val_ids = temp_df['id'].tolist()
        
    train_ids = train_df['id'].tolist()
    
    print(f"Split Statistics:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val:   {len(val_ids)} samples")
    print(f"  Test:  {len(test_ids)} samples")
    
    train_patients = set(train_df['patient_filled'])
    val_patients = set(df[df['id'].isin(val_ids)]['patient_filled'])
    intersection = train_patients.intersection(val_patients)
    if intersection:
        print(f"WARNING: Patient leakage detected! {intersection}")
    else:
        print("  No patient overlap between Train and Val.")
        
    return train_ids, val_ids, test_ids

def main():
    parser = argparse.ArgumentParser(description="Split HEST metadata into train/val/test sets by patient.")
    parser.add_argument("metadata", type=str, help="Path to HEST metadata CSV.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test set ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Error: File {args.metadata} not found.")
        return

    split_hest_patients(args.metadata, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()
