import csv
import json
import os


def filter_hest_metadata(
    csv_file, search_terms=["colon", "rectum", "colorectal", "bowel"]
):
    bowel_samples = []

    # Handle path flexibility
    if not os.path.exists(csv_file):
        csv_file_up = os.path.join("..", csv_file)
        if os.path.exists(csv_file_up):
            csv_file = csv_file_up

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            organ = row.get("organ", "").lower()
            primary_site = row.get("primary_site", "").lower()
            if any(term in organ for term in search_terms) or any(
                term in primary_site for term in search_terms
            ):
                bowel_samples.append(row["id"])

    # Also include the metadata folder and other root files
    patterns = ["HEST_v1_3_0.csv", "README.md", ".gitattributes"]
    for sample_id in bowel_samples:
        patterns.append(f"data/{sample_id}/*")
        patterns.append(f"metadata/{sample_id}.json")
        patterns.append(f"cellvit_seg/{sample_id}_cellvit_seg.parquet")

    return patterns


if __name__ == "__main__":
    patterns = filter_hest_metadata("HEST_v1_3_0.csv")
    print(json.dumps(patterns))
