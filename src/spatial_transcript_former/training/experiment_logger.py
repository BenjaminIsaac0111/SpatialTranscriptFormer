"""
Experiment Logger for offline HPC training.

Handles structured logging to CSV and JSON for experiment tracking
without requiring network access (e.g., no wandb dependency).
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Logs training metrics to a SQLite database and writes a JSON summary at the end.

    Output files:
        - training_logs.sqlite: Per-epoch metrics (epoch, train_loss, val_loss, ...) stored in a table `metrics`.
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
        self.db_path = os.path.join(output_dir, "training_logs.sqlite")
        self.json_path = os.path.join(output_dir, "results_summary.json")
        self.start_time = time.time()
        self.epoch_metrics = []

        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and metric table if it doesn't exist."""
        # Using connect as a context manager ensures commits
        with sqlite3.connect(self.db_path) as conn:
            # We use a dynamic schema where columns are added as needed.
            # Start with just 'epoch' as the primary key.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    epoch INTEGER PRIMARY KEY
                )
                """)

    def _ensure_columns(self, metrics: Dict[str, float]):
        """Ensures all metric keys exist as columns in the metrics table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(metrics)")
            existing_columns = {col[1] for col in cursor.fetchall()}

            for key in metrics.keys():
                if key not in existing_columns:
                    # SQLite alters don't fail if concurrent unless locked.
                    # Try to add missing column as REAL (float)
                    try:
                        cursor.execute(f"ALTER TABLE metrics ADD COLUMN {key} REAL")
                    except sqlite3.OperationalError:
                        # Might have been added by another process if we are running distributed
                        pass

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Insert one row into training_logs.sqlite -> metrics table.

        Args:
            epoch: Current epoch number (1-indexed).
            metrics: Dict of metric name -> value, e.g. {"train_loss": 0.1, "val_loss": 0.2}.
        """
        row = {"epoch": epoch, **metrics}
        self.epoch_metrics.append(row)

        # Ensure all columns exist before inserting
        self._ensure_columns(metrics)

        columns = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        values = tuple(row.values())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO metrics ({columns}) VALUES ({placeholders})",
                values,
            )

    def finalize(
        self,
        best_val_loss: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
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
