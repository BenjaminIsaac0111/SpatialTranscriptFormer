import os
import pandas as pd
import sqlite3
import argparse


def migrate_csv_to_sqlite(run_dir):
    csv_path = os.path.join(run_dir, "training_log.csv")
    db_path = os.path.join(run_dir, "training_logs.sqlite")

    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}")
        return

    print(f"Migrating {csv_path} to {db_path}...")
    df = pd.read_csv(csv_path)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("metrics", conn, if_exists="replace", index=False)
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to run directory"
    )
    args = parser.parse_args()
    migrate_csv_to_sqlite(args.run_dir)
