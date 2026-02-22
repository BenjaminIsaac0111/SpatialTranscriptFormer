import pandas as pd
import os


def estimate_size():
    # Load metadata
    csv_path = "HEST_v1_3_0.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("..", "HEST_v1_3_0.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    bowel_ids = set(df[df["organ"] == "Bowel"]["id"].unique())

    total_size = 0
    count = 0

    repo_files_path = "repo_files.txt"
    if not os.path.exists(repo_files_path):
        repo_files_path = os.path.join("..", "repo_files.txt")

    # Check if repo_files.txt exists
    if not os.path.exists(repo_files_path):
        print("repo_files.txt not found.")
        return

    # Read repo_files.txt (UTF-16 encoded)
    try:
        with open(repo_files_path, "r", encoding="utf-16le") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        size = int(parts[0])
                        path = parts[1]
                        # Check if any bowel ID is in the path
                        for bid in bowel_ids:
                            if bid in path:
                                total_size += size
                                count += 1
                                break
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error reading repo_files.txt: {e}")
        return

    print(f"Total Size: {total_size / (1024**3):.2f} GB")
    print(f"Total Files: {count}")


if __name__ == "__main__":
    estimate_size()
