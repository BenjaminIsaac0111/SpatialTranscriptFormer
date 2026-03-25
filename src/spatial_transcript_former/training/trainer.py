"""
High-level Trainer for SpatialTranscriptFormer.

Wraps the low-level :func:`train_one_epoch` / :func:`validate` engine with
LR scheduling, checkpointing, experiment logging, and early stopping.

Example::

    from spatial_transcript_former import SpatialTranscriptFormer
    from spatial_transcript_former.training import Trainer
    from spatial_transcript_former.training.losses import CompositeLoss

    model = SpatialTranscriptFormer(num_genes=460, backbone_name="phikon", ...)
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        criterion=CompositeLoss(),
        epochs=100,
    )
    results = trainer.fit()
    trainer.save_pretrained("./release/v1/", gene_names=my_genes)
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.optim as optim

from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.experiment_logger import ExperimentLogger
from spatial_transcript_former.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class TrainerCallback:
    """Base class for Trainer callbacks.

    Override any of these hooks.  All methods are no-ops by default.
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of :meth:`Trainer.fit`."""

    def on_train_end(self, trainer: "Trainer", results: dict) -> None:
        """Called at the end of :meth:`Trainer.fit`."""

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict) -> None:
        """Called after validation.  ``metrics`` has train_loss, val_loss, etc."""

    def should_stop(self, trainer: "Trainer", epoch: int, metrics: dict) -> bool:
        """Return ``True`` to request early stopping."""
        return False


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss does not improve for ``patience`` epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum decrease in val_loss to be considered an improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._wait = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._wait = 0
        else:
            self._wait += 1

    def should_stop(self, trainer, epoch, metrics):
        if self._wait >= self.patience:
            print(
                f"Early stopping: no improvement for {self.patience} epochs "
                f"(best={self._best_loss:.4f})."
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """High-level training orchestrator.

    Manages the full lifecycle:  LR scheduling, gradient accumulation,
    AMP, checkpointing, logging, and callbacks.

    Args:
        model: The model to train (any ``nn.Module``).
        train_loader: Training ``DataLoader``.
        val_loader: Validation ``DataLoader``.
        criterion: Loss function.
        optimizer: Optimizer.  If ``None``, ``AdamW`` is created with ``lr``
            and ``weight_decay``.
        lr: Learning rate (used only when ``optimizer`` is ``None``).
        weight_decay: Weight decay (used only when ``optimizer`` is ``None``).
        epochs: Total training epochs.
        warmup_epochs: Linear warmup epochs before cosine annealing.
        device: Device string (``"cuda"``, ``"cpu"``).
        output_dir: Directory for checkpoints and logs.
        model_name: Name used in checkpoint filenames.
        use_amp: Enable automatic mixed precision (FP16).
        grad_accum_steps: Gradient accumulation steps.
        whole_slide: Whole-slide prediction mode (training).
        val_whole_slide: Whole-slide mode for validation.  Defaults to
            ``whole_slide``.  Set to ``True`` to get proper per-slide
            spatial PCC even when training in patch mode.
        callbacks: List of :class:`TrainerCallback` instances.
        resume: Attempt to resume from a checkpoint in ``output_dir``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        epochs: int = 100,
        warmup_epochs: int = 10,
        device: str = "cuda",
        output_dir: str = "./checkpoints",
        model_name: str = "model",
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        whole_slide: bool = False,
        val_whole_slide: Optional[bool] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        resume: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.output_dir = output_dir
        self.model_name = model_name
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.whole_slide = whole_slide
        self.val_whole_slide = val_whole_slide if val_whole_slide is not None else whole_slide
        self.callbacks = callbacks or []
        self.resume = resume

        # State
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.history: List[Dict[str, Any]] = []

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        # LR Scheduler: warmup → cosine
        self._build_scheduler()

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Logger
        os.makedirs(output_dir, exist_ok=True)
        self.logger = ExperimentLogger(
            output_dir,
            {
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "use_amp": use_amp,
                "grad_accum_steps": grad_accum_steps,
                "whole_slide": whole_slide,
                "model_name": model_name,
            },
        )

        # Resume
        if resume:
            self._resume_from_checkpoint()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scheduler(self):
        warmup = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=max(1, self.warmup_epochs),
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs - self.warmup_epochs),
            eta_min=1e-6,
        )

        if self.warmup_epochs > 0:
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = cosine

    def _resume_from_checkpoint(self):
        schedulers = {"main": self.scheduler}
        start_epoch, best_val_loss, loaded_schedulers = load_checkpoint(
            self.model,
            self.optimizer,
            self.scaler,
            schedulers,
            self.output_dir,
            self.model_name,
            self.device,
        )
        self.current_epoch = start_epoch
        self.best_val_loss = best_val_loss

        # Catch up scheduler for old checkpoints
        if start_epoch > 0 and self.scheduler.last_epoch < start_epoch:
            for _ in range(start_epoch):
                self.scheduler.step()

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            dict: Final training results including ``best_val_loss`` and
            ``history`` (list of per-epoch metrics).
        """
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # --- Train ---
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                whole_slide=self.whole_slide,
                scaler=self.scaler,
                grad_accum_steps=self.grad_accum_steps,
            )

            # --- Validate ---
            val_metrics = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                whole_slide=self.val_whole_slide,
                use_amp=self.use_amp,
            )

            val_loss = val_metrics["val_loss"]
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {lr:.2e}"
            )

            # Step scheduler
            self.scheduler.step()

            # --- Metrics ---
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            }
            for key in ("val_mae", "val_pcc", "pred_variance", "attn_correlation"):
                if val_metrics.get(key) is not None:
                    epoch_metrics[key] = val_metrics[key]

            # Hardware metrics (optional)
            try:
                import psutil

                epoch_metrics["sys_cpu_percent"] = psutil.cpu_percent()
                epoch_metrics["sys_ram_percent"] = psutil.virtual_memory().percent
            except ImportError:
                pass

            if torch.cuda.is_available():
                epoch_metrics["sys_gpu_mem_mb"] = round(
                    torch.cuda.memory_allocated() / (1024**2), 2
                )

            self.history.append(epoch_metrics)
            self.logger.log_epoch(epoch + 1, epoch_metrics)

            # --- Best model ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(
                    self.output_dir, f"best_model_{self.model_name}.pth"
                )
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best model -> {best_path}")

            # --- Checkpoint ---
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scaler,
                {"main": self.scheduler},
                epoch,
                self.best_val_loss,
                self.output_dir,
                self.model_name,
            )

            # --- Callbacks ---
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, epoch_metrics)

            if any(cb.should_stop(self, epoch, epoch_metrics) for cb in self.callbacks):
                print(f"Training stopped at epoch {epoch + 1}.")
                break

        # --- Finalize ---
        results = {
            "best_val_loss": self.best_val_loss,
            "epochs_completed": self.current_epoch + 1,
            "history": self.history,
        }

        self.logger.finalize(self.best_val_loss)

        for cb in self.callbacks:
            cb.on_train_end(self, results)

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def save_pretrained(
        self, path: str, gene_names: Optional[List[str]] = None
    ) -> None:
        """Export an inference-ready checkpoint (strips optimizer state).

        Delegates to :func:`spatial_transcript_former.checkpoint.save_pretrained`.
        """
        from spatial_transcript_former.checkpoint import (
            save_pretrained as _save_pretrained,
        )

        _save_pretrained(self.model, path, gene_names=gene_names)
