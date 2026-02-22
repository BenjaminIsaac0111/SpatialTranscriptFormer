"""
Experiment Logger for offline HPC training.

Handles structured logging to CSV and JSON for experiment tracking
without requiring network access (e.g., no wandb dependency).
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Logs training metrics to CSV and writes a JSON summary at the end.

    Output files:
        - training_log.csv: Per-epoch metrics (epoch, train_loss, val_loss, ...)
        - results_summary.json: Full config + final metrics
    """

    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Args:
            output_dir: Directory to write log files to.
            config: Dictionary of all experiment hyperparameters (from argparse).
        """
        self.output_dir = output_dir
        self.config = config
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.json_path = os.path.join(output_dir, "results_summary.json")
        self.start_time = time.time()
        self.epoch_metrics = []
        self._csv_header_written = os.path.exists(self.csv_path)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Append one row to training_log.csv.

        Args:
            epoch: Current epoch number (1-indexed).
            metrics: Dict of metric name -> value, e.g. {"train_loss": 0.1, "val_loss": 0.2}.
        """
        row = {"epoch": epoch, **metrics}
        self.epoch_metrics.append(row)

        # Determine fieldnames from first row
        fieldnames = list(row.keys())

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def finalize(
        self, best_val_loss: float, extra_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Write results_summary.json with full experiment metadata.

        Args:
            best_val_loss: Best validation loss achieved.
            extra_metrics: Any additional metrics to include (e.g., attn_correlation).
        """
        elapsed = time.time() - self.start_time

        summary = {
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(elapsed, 1),
            "epochs_completed": len(self.epoch_metrics),
            "best_val_loss": round(best_val_loss, 6),
            "config": self.config,
        }

        # Add final epoch metrics
        if self.epoch_metrics:
            summary["final_epoch"] = self.epoch_metrics[-1]

        if extra_metrics:
            summary.update(extra_metrics)

        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Results summary saved to {self.json_path}")
