#!/usr/bin/env python
"""
Unified comparative evaluation script.
Trains all baseline models and SpatialTranscriptFormer from scratch under the
same data splits, aggregates their final performance/computational metrics,
and generates formatted markdown summaries and performance charts.
"""

import os
import sys
import json
import sqlite3
import subprocess
import argparse
import time
import pathlib
from typing import Dict, Any, List


# Curated list of CRC pathways
CRC_PATHWAYS = [
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_KRAS_SIGNALING_DN",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_APICAL_JUNCTION",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_HYPOXIA",
]


def run_experiment(model_name: str, cmd_args: List[str]) -> bool:
    """Run a single training experiment as an isolated subprocess."""
    print(f"\n========================================================")
    print(f"Starting experiment for: {model_name}")
    print(
        f"Command: {sys.executable} -m spatial_transcript_former.train {' '.join(cmd_args)}"
    )
    print(f"========================================================\n")

    cmd = [sys.executable, "-m", "spatial_transcript_former.train"] + cmd_args
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Experiment {model_name} failed with exit code {e.returncode}")
        return False


def collect_metrics(output_dir: str) -> Dict[str, Any]:
    """Retrieve metrics from SQLite logs and json summary for the best epoch."""
    metrics = {
        "best_epoch": None,
        "val_loss": None,
        "val_mae": None,
        "val_pcc": None,
        "val_ccc": None,
        "pred_variance": None,
        "spatial_coherence": None,
        "runtime_seconds": None,
        "sys_gpu_mem_mb": None,
    }

    db_path = os.path.join(output_dir, "training_logs.sqlite")
    json_path = os.path.join(output_dir, "results_summary.json")

    # Read SQLite metrics for the best validation CCC epoch
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(metrics)")
            cols = [c[1] for c in cur.fetchall()]

            if "val_ccc" in cols:
                # Retrieve the epoch with highest val_ccc
                cur.execute("SELECT * FROM metrics ORDER BY val_ccc DESC LIMIT 1")
            else:
                cur.execute("SELECT * FROM metrics ORDER BY val_loss ASC LIMIT 1")

            row = cur.fetchone()
            if row:
                row_dict = dict(zip(cols, row))
                metrics["best_epoch"] = row_dict.get("epoch")
                metrics["val_loss"] = row_dict.get("val_loss")
                metrics["val_mae"] = row_dict.get("val_mae")
                metrics["val_pcc"] = row_dict.get("val_pcc")
                metrics["val_ccc"] = row_dict.get("val_ccc")
                metrics["pred_variance"] = row_dict.get("pred_variance")
                metrics["spatial_coherence"] = row_dict.get("spatial_coherence")
                metrics["sys_gpu_mem_mb"] = row_dict.get("sys_gpu_mem_mb")
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to read metrics from SQLite database: {e}")

    # Read runtime from JSON summary
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                summary = json.load(f)
                metrics["runtime_seconds"] = summary.get("runtime_seconds")
        except Exception as e:
            print(f"Warning: Failed to read summary from JSON: {e}")

    return metrics


def build_markdown_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format experimental results as a markdown table."""
    headers = [
        "Model",
        "Best Epoch",
        "Val Loss",
        "Val MAE",
        "Val PCC",
        "Val CCC",
        "Pred Var",
        "Spatial Coherence",
        "Runtime (s)",
        "Peak VRAM (MB)",
    ]

    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model_name, m in results.items():
        row = [
            f"**{model_name}**",
            str(m.get("best_epoch") or "N/A"),
            f"{m.get('val_loss'):.4f}" if m.get("val_loss") is not None else "N/A",
            f"{m.get('val_mae'):.4f}" if m.get("val_mae") is not None else "N/A",
            f"{m.get('val_pcc'):.4f}" if m.get("val_pcc") is not None else "N/A",
            f"{m.get('val_ccc'):.4f}" if m.get("val_ccc") is not None else "N/A",
            (
                f"{m.get('pred_variance'):.6f}"
                if m.get("pred_variance") is not None
                else "N/A"
            ),
            (
                f"{m.get('spatial_coherence'):.4f}"
                if m.get("spatial_coherence") is not None
                else "N/A"
            ),
            (
                f"{m.get('runtime_seconds'):.1f}"
                if m.get("runtime_seconds") is not None
                else "N/A"
            ),
            (
                f"{m.get('sys_gpu_mem_mb'):.1f}"
                if m.get("sys_gpu_mem_mb") is not None
                else "N/A"
            ),
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def plot_charts(results: Dict[str, Dict[str, Any]], output_path: str):
    """Generate comparative performance charts using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(results.keys())
        pcc_vals = [r.get("val_pcc") or 0.0 for r in results.values()]
        ccc_vals = [r.get("val_ccc") or 0.0 for r in results.values()]
        mae_vals = [r.get("val_mae") or 0.0 for r in results.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # PCC vs CCC Bar Chart
        x = range(len(models))
        width = 0.35
        ax1.bar(
            [i - width / 2 for i in x],
            pcc_vals,
            width,
            label="PCC (Correlation)",
            color="#3498db",
        )
        ax1.bar(
            [i + width / 2 for i in x],
            ccc_vals,
            width,
            label="CCC (Concordance)",
            color="#2ecc71",
        )
        ax1.set_ylabel("Score")
        ax1.set_title("Spatially-Resolved Pathway Correlation (PCC vs. CCC)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15)
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # MAE Bar Chart
        ax2.bar(models, mae_vals, 0.5, color="#e74c3c")
        ax2.set_ylabel("MAE (Lower is Better)")
        ax2.set_title("Absolute Predictive Error (MAE)")
        ax2.set_xticklabels(models, rotation=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Comparison charts saved to: {output_path}")
    except Exception as e:
        print(
            f"Warning: Failed to generate charts (matplotlib may be missing or failed): {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run complete model comparative experiments"
    )
    parser.add_argument(
        "--data-dir", type=str, default="A:\\hest_data", help="HEST data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/comparison",
        help="Output runs directory",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick verification (1 epoch, 2 samples)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split repeatability"
    )
    parser.add_argument(
        "--epochs-patch",
        type=int,
        default=5,
        help="Number of training epochs for patch baselines",
    )
    parser.add_argument(
        "--epochs-mil",
        type=int,
        default=20,
        help="Number of training epochs for MIL/STF models",
    )

    # Slurm & Execution Options
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate Slurm scripts instead of running locally",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated Slurm jobs immediately (requires --slurm)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect results and generate report/charts without running training",
    )

    # Slurm Resource Options
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="gpu",
        help="Slurm partition for training jobs",
    )
    parser.add_argument(
        "--slurm-collect-partition",
        type=str,
        default=None,
        help="Slurm partition for the collection job (defaults to cluster default)",
    )
    parser.add_argument(
        "--slurm-gres", type=str, default="gpu:1", help="Slurm GRES for training jobs"
    )
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="12:00:00",
        help="Slurm time limit for training jobs",
    )
    parser.add_argument(
        "--slurm-mem",
        type=str,
        default="32G",
        help="Slurm memory limit for training jobs",
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=4,
        help="Slurm CPUs per task for training jobs",
    )

    # Slurm Environment Setup
    parser.add_argument(
        "--slurm-conda",
        type=str,
        default="SpatialTranscriptFormer",
        help="Conda environment to activate in Slurm jobs",
    )
    parser.add_argument(
        "--slurm-setup-cmds",
        type=str,
        default=None,
        help="Raw bash commands to run before the python command (e.g. 'module load cuda'). "
        "If not specified, defaults to activating the conda environment specified by --slurm-conda.",
    )
    args = parser.parse_args()

    # Determine hyperparameters based on --quick-test
    max_samples = 5 if args.quick_test else None
    epochs_patch = 1 if args.quick_test else args.epochs_patch
    epochs_mil = 1 if args.quick_test else args.epochs_mil

    # Configurations for each model
    configs = {
        "HE2RNA": [
            "--model",
            "he2rna",
            "--backbone",
            "resnet50",
            "--batch-size",
            "64",
            "--epochs",
            str(epochs_patch),
            "--warmup-epochs",
            "0" if args.quick_test else "2",
        ],
        "ViT_ST": [
            "--model",
            "vit_st",
            "--backbone",
            "vit_b_16",
            "--batch-size",
            "32",
            "--epochs",
            str(epochs_patch),
            "--warmup-epochs",
            "0" if args.quick_test else "2",
        ],
        "AttentionMIL": [
            "--model",
            "attention_mil",
            "--backbone",
            "ctranspath",
            "--whole-slide",
            "--precomputed",
            "--weak-supervision",
            "--use-amp",
            "--batch-size",
            "1",
            "--epochs",
            str(epochs_mil),
            "--warmup-epochs",
            "0" if args.quick_test else "4",
        ],
        "TransMIL": [
            "--model",
            "transmil",
            "--backbone",
            "ctranspath",
            "--whole-slide",
            "--precomputed",
            "--weak-supervision",
            "--use-amp",
            "--batch-size",
            "1",
            "--epochs",
            str(epochs_mil),
            "--warmup-epochs",
            "0" if args.quick_test else "4",
        ],
        "SpatialTranscriptFormer": [
            "--model",
            "interaction",
            "--backbone",
            "ctranspath",
            "--whole-slide",
            "--precomputed",
            "--use-amp",
            "--batch-size",
            "8",
            "--token-dim",
            "512",
            "--n-heads",
            "8",
            "--n-layers",
            "6",
            "--loss",
            "mse_ccc",
            "--epochs",
            str(epochs_mil),
            "--warmup-epochs",
            "0" if args.quick_test else "4",
        ],
    }

    # Add shared flags to all configs
    for name, cmd in configs.items():
        cmd.extend(
            [
                "--data-dir",
                args.data_dir,
                "--organ",
                "Bowel",
                "--seed",
                str(args.seed),
                "--output-dir",
                os.path.join(args.output_dir, name.lower()),
                "--pathways",
            ]
            + CRC_PATHWAYS
        )

        if max_samples:
            cmd.extend(["--max-samples", str(max_samples)])

    results = {}

    if args.collect_only:
        print(f"\n--- Running in Collection Mode (Reading from {args.output_dir}) ---")
        for model_name in configs.keys():
            out_dir = os.path.join(args.output_dir, model_name.lower())
            db_path = os.path.join(out_dir, "training_logs.sqlite")
            json_path = os.path.join(out_dir, "results_summary.json")
            if os.path.exists(db_path) or os.path.exists(json_path):
                print(f"Collecting metrics for: {model_name}")
                results[model_name] = collect_metrics(out_dir)
            else:
                print(f"No results found for model: {model_name} in {out_dir}")
    elif args.slurm:
        slurm_scripts_dir = os.path.join(args.output_dir, "slurm_scripts")
        os.makedirs(slurm_scripts_dir, exist_ok=True)

        if args.slurm_setup_cmds:
            setup_cmds = args.slurm_setup_cmds
        else:
            setup_cmds = f"source $(conda info --base)/etc/profile.d/conda.sh\nconda activate {args.slurm_conda}"

        job_ids = []
        slurm_script_paths = {}
        working_dir = pathlib.Path(os.getcwd()).as_posix()

        print(f"\nGenerating Slurm scripts in: {slurm_scripts_dir}")

        for model_name, cmd_args in configs.items():
            posix_cmd_args = []
            for arg in cmd_args:
                # Convert backslashes to forward slashes for Unix compatibility
                if (
                    isinstance(arg, str)
                    and ("\\" in arg or "/" in arg)
                    and not arg.startswith("--")
                ):
                    posix_cmd_args.append(pathlib.Path(arg).as_posix())
                else:
                    posix_cmd_args.append(str(arg))

            model_out_dir = pathlib.Path(
                os.path.join(args.output_dir, model_name.lower())
            ).as_posix()
            log_path = f"{model_out_dir}/{model_name.lower()}_slurm.log"

            # Ensure model output directory exists
            os.makedirs(
                os.path.join(args.output_dir, model_name.lower()), exist_ok=True
            )

            script_content = f"""#!/bin/bash
#SBATCH --job-name=stf_compare_{model_name.lower()}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition={args.slurm_partition}
#SBATCH --gres={args.slurm_gres}
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}

{setup_cmds}

# Change to submit directory (project root)
cd "${{SLURM_SUBMIT_DIR:-.}}"

python -m spatial_transcript_former.train {" ".join(posix_cmd_args)}
"""
            script_filename = f"{model_name.lower()}.slurm"
            script_path = os.path.join(slurm_scripts_dir, script_filename)
            with open(script_path, "w", newline="\n") as f:
                f.write(script_content)
            slurm_script_paths[model_name] = script_path
            print(f"  - Generated {script_filename}")

        # Generate collection slurm script
        posix_output_dir = pathlib.Path(args.output_dir).as_posix()
        collect_log_path = f"{posix_output_dir}/collect_slurm.log"
        collect_partition_line = (
            f"#SBATCH --partition={args.slurm_collect_partition}\n"
            if args.slurm_collect_partition
            else ""
        )
        collect_script_content = f"""#!/bin/bash
#SBATCH --job-name=stf_compare_collect
#SBATCH --output={collect_log_path}
#SBATCH --error={collect_log_path}
{collect_partition_line}#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

{setup_cmds}

# Change to submit directory (project root)
cd "${{SLURM_SUBMIT_DIR:-.}}"

python scripts/compare_models.py --collect-only --output-dir {posix_output_dir}
"""
        collect_script_path = os.path.join(slurm_scripts_dir, "collect.slurm")
        with open(collect_script_path, "w", newline="\n") as f:
            f.write(collect_script_content)
        print("  - Generated collect.slurm")

        if args.submit:
            print("\nSubmitting jobs to Slurm scheduler...")
            for model_name, path in slurm_script_paths.items():
                try:
                    cmd = ["sbatch", path]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    output = result.stdout.strip()
                    print(f"  Submitted {model_name} job: {output}")
                    job_id = output.split()[-1]
                    job_ids.append(job_id)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"  Error submitting job for {model_name}: {e}")
                    print(
                        "  Make sure 'sbatch' is installed and available in your PATH."
                    )

            if job_ids:
                dependency_str = f"--dependency=afterany:{':'.join(job_ids)}"
                try:
                    cmd = ["sbatch", dependency_str, collect_script_path]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        f"  Submitted collection job with dependency: {result.stdout.strip()}"
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"  Error submitting collection job: {e}")
            else:
                print(
                    "  No jobs were submitted successfully, skipping collection job submission."
                )
        else:
            print("\nTo run the comparison on Slurm, submit the generated scripts:")
            for model_name, path in slurm_script_paths.items():
                posix_path = pathlib.Path(path).as_posix()
                print(f"  sbatch {posix_path}")
            posix_collect_path = pathlib.Path(collect_script_path).as_posix()
            print(f"\nAfter all jobs complete, run the collection script manually:")
            print(f"  sbatch {posix_collect_path}")
            print(f"Or run the comparison script directly in collection mode:")
            print(
                f"  python scripts/compare_models.py --collect-only --output-dir {posix_output_dir}"
            )

        return
    else:
        # Run each training experiment sequentially (original local behavior)
        for model_name, cmd_args in configs.items():
            success = run_experiment(model_name, cmd_args)
            if success:
                out_dir = os.path.join(args.output_dir, model_name.lower())
                results[model_name] = collect_metrics(out_dir)
            else:
                print(f"Skipping metric collection for failed run: {model_name}")

    # Print results summary
    if results:
        print("\n\n" + "=" * 60)
        print("EXPERIMENTAL COMPARISON RESULTS")
        print("=" * 60)
        table_md = build_markdown_table(results)
        print(table_md)
        print("=" * 60 + "\n")

        # Save markdown report
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("# Comparative Model Performance Evaluation (Bowel/CRC subset)\n\n")
            f.write(table_md)
            f.write(
                "\n\n*Note: Metrics correspond to the epoch that achieved the best Concordance Correlation Coefficient (CCC) or lowest Validation Loss.*"
            )
        print(f"Saved report summary to: {report_path}")

        # Save CSV results
        csv_path = os.path.join(args.output_dir, "comparison_results.csv")
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Headers
            writer.writerow(
                [
                    "Model",
                    "BestEpoch",
                    "ValLoss",
                    "ValMAE",
                    "ValPCC",
                    "ValCCC",
                    "PredVar",
                    "SpatialCoherence",
                    "RuntimeSeconds",
                    "PeakGPUMemMB",
                ]
            )
            for m_name, m in results.items():
                writer.writerow(
                    [
                        m_name,
                        m.get("best_epoch"),
                        m.get("val_loss"),
                        m.get("val_mae"),
                        m.get("val_pcc"),
                        m.get("val_ccc"),
                        m.get("pred_variance"),
                        m.get("spatial_coherence"),
                        m.get("runtime_seconds"),
                        m.get("sys_gpu_mem_mb"),
                    ]
                )
        print(f"Saved CSV results to: {csv_path}")

        # Generate comparative charts
        chart_path = os.path.join(args.output_dir, "comparison_chart.png")
        plot_charts(results, chart_path)

    else:
        print("\nNo experiments completed successfully. No results generated.\n")


if __name__ == "__main__":
    main()
