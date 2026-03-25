import os
import sys
sys.path.append(r"z:\Projects\SpatialTranscriptFormer\scripts")
from batch_qc_stats import calculate_qc_stats
import numpy as np

st_dir = r"A:\hest_data\st"
samples = [f.replace(".h5ad", "") for f in os.listdir(st_dir) if f.endswith(".h5ad")]
samples.sort()

results = []
for s in samples:
    try:
        total, kept, l_umi, l_gene, h_mt = calculate_qc_stats(os.path.join(st_dir, f"{s}.h5ad"))
        results.append({
            "sample": s,
            "total": total,
            "kept": kept,
            "pct": kept / total if total > 0 else 0,
            "low_umi": l_umi,
            "low_gene": l_gene,
            "high_mt": h_mt
        })
    except Exception as e:
        print(f"Error {s}: {e}")

results.sort(key=lambda x: x["pct"])

print(f"{'Sample':<15} | {'Kept %':<8} | {'Filtered':<10} | {'Low UMI':<8} | {'Low Gene':<8} | {'High MT':<8}")
print("-" * 75)
for r in results[:15]:
    print(f"{r['sample']:<15} | {r['pct']:7.1%} | {r['total']-r['kept']:<10} | {r['low_umi']:<8} | {r['low_gene']:<8} | {r['high_mt']:<8}")
