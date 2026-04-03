"""
Tests for Trainer lifecycle, callbacks, and checkpoint resumption.
"""

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spatial_transcript_former.training.trainer import (
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from spatial_transcript_former.data.base import SpatialDataset
from spatial_transcript_former.recipes.hest.dataset import collate_fn_patch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TinySpatialDataset(SpatialDataset):
    """Minimal SpatialDataset for testing."""

    def __init__(self, n=32, feature_dim=64, num_pathways=10):
        self._features = torch.randn(n, 1, feature_dim)
        self._pathways = torch.randn(n, num_pathways).abs()
        self._coords = torch.zeros(n, 1, 2)
        self.num_pathways = num_pathways

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        return self._features[idx], None, self._pathways[idx], self._coords[idx]


class TinyModel(nn.Module):
    """Simple linear model for testing (mimics patch-level prediction)."""

    def __init__(self, in_dim=64, num_pathways=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_pathways)

    def forward(self, x, **kwargs):
        # x shape: (B, 1, D) -> squeeze -> (B, D)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.fc(x)


@pytest.fixture
def tiny_setup(tmp_path):
    """Create a minimal training setup with a tmp_path for output."""
    ds = TinySpatialDataset(n=32, feature_dim=64, num_pathways=10)
    train_loader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn=collate_fn_patch
    )
    val_loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn_patch)

    model = TinyModel(in_dim=64, num_pathways=10)
    criterion = nn.MSELoss()
    output_dir = str(tmp_path / "output")

    return model, criterion, train_loader, val_loader, output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainerImports:
    """Verify the Trainer is importable from all expected paths."""

    def test_top_level_import(self):
        from spatial_transcript_former import Trainer

        assert Trainer is not None

    def test_training_subpackage_import(self):
        from spatial_transcript_former.training import Trainer

        assert Trainer is not None

    def test_direct_import(self):
        from spatial_transcript_former.training.trainer import Trainer

        assert Trainer is not None

    def test_callback_imports(self):
        from spatial_transcript_former.training import (
            TrainerCallback,
            EarlyStoppingCallback,
        )

        assert TrainerCallback is not None
        assert EarlyStoppingCallback is not None


class TestTrainerBasicFit:
    """Test the core fit() lifecycle."""

    def test_fit_runs_to_completion(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=3,
            warmup_epochs=1,
            device="cpu",
            output_dir=output_dir,
            use_amp=False,
        )

        results = trainer.fit()

        assert "best_val_loss" in results
        assert results["epochs_completed"] == 3
        assert len(results["history"]) == 3

    def test_fit_records_metrics(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=2,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
        )
        results = trainer.fit()

        for row in results["history"]:
            assert "train_loss" in row
            assert "val_loss" in row
            assert "lr" in row

    def test_saves_best_model_and_checkpoint(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=2,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
            model_name="test",
        )
        trainer.fit()

        assert os.path.exists(os.path.join(output_dir, "best_model_test.pth"))
        assert os.path.exists(os.path.join(output_dir, "latest_model_test.pth"))

    def test_saves_logger_outputs(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=2,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
        )
        trainer.fit()

        assert os.path.exists(os.path.join(output_dir, "training_logs.sqlite"))
        assert os.path.exists(os.path.join(output_dir, "results_summary.json"))


class TestTrainerCustomOptimizer:
    """Test passing a custom optimizer."""

    def test_custom_optimizer(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=2,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
        )
        assert trainer.optimizer is optimizer
        results = trainer.fit()
        assert results["epochs_completed"] == 2


class TestCallbacks:
    """Test the callback system."""

    def test_callbacks_are_called(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        call_log = []

        class LogCallback(TrainerCallback):
            def on_train_begin(self, trainer):
                call_log.append("train_begin")

            def on_epoch_begin(self, trainer, epoch):
                call_log.append(f"epoch_begin_{epoch}")

            def on_epoch_end(self, trainer, epoch, metrics):
                call_log.append(f"epoch_end_{epoch}")

            def on_train_end(self, trainer, results):
                call_log.append("train_end")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=2,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
            callbacks=[LogCallback()],
        )
        trainer.fit()

        assert call_log == [
            "train_begin",
            "epoch_begin_0",
            "epoch_end_0",
            "epoch_begin_1",
            "epoch_end_1",
            "train_end",
        ]

    def test_early_stopping(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        # Wrap so the model always outputs a near-constant â†’ loss stays flat â†’ early stop
        class ConstantModel(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base  # keep parameters so optimizer doesn't crash

            def forward(self, x, **kwargs):
                out = self.base(x, **kwargs)
                # Multiply by tiny eps to keep grad flow, but output is ~1.0
                return out * 1e-10 + 1.0

        const_model = ConstantModel(model)

        trainer = Trainer(
            model=const_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=100,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
            callbacks=[EarlyStoppingCallback(patience=2)],
        )
        results = trainer.fit()

        # With constant output, loss never changes, so early stopping should
        # fire after patience + 1 epochs
        assert results["epochs_completed"] <= 5


class TestTrainerResume:
    """Test checkpoint resumption."""

    def test_resume_continues_from_checkpoint(self, tiny_setup):
        model, criterion, train_loader, val_loader, output_dir = tiny_setup

        # Train for 3 epochs
        trainer1 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=3,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
            model_name="resume_test",
        )
        trainer1.fit()

        # Resume â€” should start from epoch 3 and run to 5
        trainer2 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=5,
            warmup_epochs=0,
            device="cpu",
            output_dir=output_dir,
            model_name="resume_test",
            resume=True,
        )
        results2 = trainer2.fit()

        # Should have completed 5 total epochs (3 from first run + 2 more)
        assert results2["epochs_completed"] == 5
