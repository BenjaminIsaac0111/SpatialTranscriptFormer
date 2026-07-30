import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from typing import Iterator, List, Optional, Tuple, Union

import argparse


def split_hest_patients(
    metadata_path: str, val_ratio: float = 0.2, test_ratio: float = 0.0, seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits HEST samples into train/val/test based on Patient ID to prevent data leakage.
    Samples with missing patient IDs are treated as unique patients (safe fallback).
    """
    df = pd.read_csv(metadata_path)
    df["patient_filled"] = df["patient"].apply(
        lambda x: None if pd.isna(x) or str(x).strip() == "" else x
    )
    df["patient_filled"] = df["patient_filled"].fillna(df["id"])

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio + test_ratio, random_state=seed
    )
    train_idx, temp_idx = next(splitter.split(df, groups=df["patient_filled"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    val_ids = []
    test_ids = []

    if test_ratio > 0:
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        test_relative_ratio = 1.0 - val_relative_ratio

        if len(temp_df["patient_filled"].unique()) > 1:
            splitter_2 = GroupShuffleSplit(
                n_splits=1, test_size=test_relative_ratio, random_state=seed
            )
            val_idx, test_idx = next(
                splitter_2.split(temp_df, groups=temp_df["patient_filled"])
            )
            val_ids = temp_df.iloc[val_idx]["id"].tolist()
            test_ids = temp_df.iloc[test_idx]["id"].tolist()
        else:
            val_ids = temp_df["id"].tolist()
    else:
        val_ids = temp_df["id"].tolist()

    train_ids = train_df["id"].tolist()

    print(f"Split Statistics:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val:   {len(val_ids)} samples")
    print(f"  Test:  {len(test_ids)} samples")

    train_patients = set(train_df["patient_filled"])
    val_patients = set(df[df["id"].isin(val_ids)]["patient_filled"])
    intersection = train_patients.intersection(val_patients)
    if intersection:
        print(f"WARNING: Patient leakage detected! {intersection}")
    else:
        print("  No patient overlap between Train and Val.")

    return train_ids, val_ids, test_ids


# The dominant Bowel/Visium study (41/73 slides) — see doc §3. Used as the
# default for the COLON-MAP-vs-rest headline split.
DEFAULT_COLONMAP_TITLE = "COLON MAP: Colon Molecular Atlas Project"


def split_hest_by_dataset(
    metadata_path: str,
    held_out: Union[str, List[str]],
    organ: Optional[str] = None,
    technology: Optional[str] = None,
    species: Optional[str] = "Homo sapiens",
) -> Tuple[List[str], List[str]]:
    """Splits HEST samples by source study (``dataset_title``) to prevent
    institutional leakage, holding out one or more named studies as the
    validation set.

    Unlike ``patient`` (unusable for this corpus — see
    ``split_hest_patients`` and docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §3:
    58/73 Bowel-Visium slides have no real patient label, and where labels do
    exist they collide across different real people in different studies),
    ``dataset_title`` is populated and consistent for every row, so grouping
    on it needs no leakage-prone fallback. The split is a deterministic
    partition, not a random group-shuffle: every sample not in ``held_out``
    goes to train.

    Args:
        metadata_path: Path to the HEST metadata CSV.
        held_out: One ``dataset_title`` (or a list of them) to hold out as
            the validation set — pass a single study for a Leave-One-Study-
            Out fold, or the 9 non-headline studies for the reverse
            direction of a COLON-MAP-vs-rest-style split.
        organ: Optional ``organ`` column filter (e.g. ``"Bowel"``).
        technology: Optional ``st_technology`` column filter (e.g.
            ``"Visium"``, to exclude Xenium/Visium HD — see doc §2).
        species: Optional ``species`` column filter, default ``"Homo
            sapiens"`` (matches ``get_sample_ids``'s unconditional human-only
            filter elsewhere in the pipeline) — pass ``None`` to disable.

    Returns:
        Tuple of (train_ids, val_ids).
    """
    df = pd.read_csv(metadata_path)
    if organ:
        df = df[df["organ"] == organ]
    if technology:
        df = df[df["st_technology"] == technology]
    if species:
        df = df[df["species"] == species]

    held_out_set = {held_out} if isinstance(held_out, str) else set(held_out)
    available_studies = set(df["dataset_title"].dropna().unique())
    unknown = held_out_set - available_studies
    if unknown:
        raise ValueError(
            f"Unknown dataset_title(s) {unknown}. Available studies: "
            f"{sorted(available_studies)}"
        )

    val_mask = df["dataset_title"].isin(held_out_set)
    val_ids = df.loc[val_mask, "id"].tolist()
    train_ids = df.loc[~val_mask, "id"].tolist()

    n_train_studies = df.loc[~val_mask, "dataset_title"].nunique()
    print(
        f"Dataset-grouped split (held out {sorted(held_out_set)}): "
        f"Train {len(train_ids)} samples / {n_train_studies} studies, "
        f"Val {len(val_ids)} samples / {len(held_out_set)} studies"
    )
    return train_ids, val_ids


def iter_loso_folds(
    metadata_path: str,
    organ: Optional[str] = None,
    technology: Optional[str] = None,
    species: Optional[str] = "Homo sapiens",
) -> Iterator[Tuple[str, List[str], List[str]]]:
    """Yields one ``(held_out_study, train_ids, val_ids)`` tuple per distinct
    ``dataset_title`` — the study-grouped Leave-One-Study-Out protocol (doc
    §3). Studies are yielded in sorted-name order for reproducible fold
    ordering across runs.
    """
    df = pd.read_csv(metadata_path)
    if organ:
        df = df[df["organ"] == organ]
    if technology:
        df = df[df["st_technology"] == technology]
    if species:
        df = df[df["species"] == species]

    for study in sorted(df["dataset_title"].dropna().unique()):
        train_ids, val_ids = split_hest_by_dataset(
            metadata_path,
            held_out=study,
            organ=organ,
            technology=technology,
            species=species,
        )
        yield study, train_ids, val_ids


def split_colonmap_vs_rest(
    metadata_path: str,
    colonmap_title: str = DEFAULT_COLONMAP_TITLE,
    reverse: bool = False,
    organ: Optional[str] = None,
    technology: Optional[str] = None,
    species: Optional[str] = "Homo sapiens",
) -> Tuple[List[str], List[str]]:
    """COLON-MAP-vs-rest headline split (doc §3): train on the 9 non-
    COLON-MAP studies, evaluate on COLON MAP — the largest single LOSO fold
    and the cleanest cross-institution generalisation probe given the
    corpus's imbalance. ``reverse=True`` swaps the direction (train on
    COLON MAP, evaluate on the rest).

    Returns:
        Tuple of (train_ids, val_ids).
    """
    train_ids, val_ids = split_hest_by_dataset(
        metadata_path,
        held_out=colonmap_title,
        organ=organ,
        technology=technology,
        species=species,
    )
    return (val_ids, train_ids) if reverse else (train_ids, val_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Split HEST metadata into train/val/test sets by patient."
    )
    parser.add_argument("metadata", type=str, help="Path to HEST metadata CSV.")
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation set ratio."
    )
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test set ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"Error: File {args.metadata} not found.")
        return

    split_hest_patients(args.metadata, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
