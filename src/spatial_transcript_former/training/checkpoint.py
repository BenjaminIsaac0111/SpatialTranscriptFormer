import os
import torch


def save_checkpoint(
    model, optimizer, scaler, schedulers, epoch, best_val_loss, output_dir, model_name
):
    """Save training state for resuming."""
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        save_dict["scaler_state_dict"] = scaler.state_dict()
    if schedulers is not None:
        save_dict["schedulers_state_dict"] = {
            k: v.state_dict() for k, v in schedulers.items()
        }

    torch.save(save_dict, os.path.join(output_dir, f"latest_model_{model_name}.pth"))


def load_checkpoint(
    model, optimizer, scaler, schedulers, output_dir, model_name, device
):
    """
    Load checkpoint if it exists.

    Returns:
        tuple: (start_epoch, best_val_loss, loaded_schedulers)
    """
    ckpt_path = os.path.join(output_dir, f"latest_model_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
        return 0, float("inf"), False

    print(f"Resuming from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except (EOFError, RuntimeError, Exception) as e:
        print(
            f"Failed to load checkpoint at {ckpt_path} due to error: {e}. Starting from scratch."
        )
        return 0, float("inf"), False

    incompatible_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        print(f"Loaded with incompatible keys: {incompatible_keys}")
    try:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "schedulers_state_dict" in checkpoint and schedulers is not None:
            for k, v in checkpoint["schedulers_state_dict"].items():
                if k in schedulers:
                    schedulers[k].load_state_dict(v)
            loaded_schedulers = True
        else:
            loaded_schedulers = False
    except (ValueError, Exception) as e:
        print(
            f"Failed to load optimizer/scheduler states due to architecture change ({e}). Starting from scratch."
        )
        return 0, float("inf"), False

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resumed at epoch {start_epoch + 1}")
    return start_epoch, best_val_loss, loaded_schedulers
