import json
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils.constants import CHECKPOINT_FOLDER_NAME


def save_checkpoint(
    model: torch.nn.Module,
    model_folder_path: str,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
) -> None:
    """
    Save the model, optimizer, scheduler, and training state to a checkpoint file.

    :param model_folder_path: Path to the folder where the checkpoint will be saved.
    :param epoch: Current epoch number.
    :param best_val_loss: Best validation loss achieved so far.
    :param epochs_without_improvement: Number of epochs without improvement in validation loss.
    :param optimizer: Optimizer state to be saved.
    :param scheduler: Scheduler state to be saved.
    """
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    # Store all the training state in a dictionary to properly restore the checkpoint afterwards
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    # Save the model checkpoint
    torch.save(
        checkpoint,
        os.path.join(model_folder_path, "last_epoch_checkpoint.pth"),
    )
    with open(os.path.join(model_folder_path, "training_state.json"), "w") as f:
        f: Any
        t_state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        # Convert any ndarray objects to lists
        t_state = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in t_state.items()
        }
        json.dump(t_state, f)


def load_checkpoint(
    model: torch.nn.Module,
    model_folder_path: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    load_best: bool = False,
) -> Tuple[int, float, int]:
    """
    Load the model, optimizer, scheduler, and training state from a checkpoint file.

    :param model_folder_path: Path to the folder where the checkpoint is saved.
    :param optimizer: Optimizer to load the state into.
    :param scheduler: Scheduler to load the state into.
    :param load_best: Boolean flag to indicate whether to load the best model checkpoint.

    :return: Tuple containing the epoch number, the best validation loss, and epochs without improvement.
    """
    # If the best model must be loaded
    if load_best:
        checkpoint_folder_path = os.path.join(
            model_folder_path, CHECKPOINT_FOLDER_NAME, "best"
        )
    else:
        # Build the path to the checkpoint folder
        checkpoint_folder_path = os.path.join(model_folder_path, CHECKPOINT_FOLDER_NAME)
    # Build the path to the last epoch checkpoint file
    checkpoint_path = os.path.join(checkpoint_folder_path, "last_epoch_checkpoint.pth")
    # If a checkpoint file exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        # Load the model's parameters at the last checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        # Load the epoch number, the best validation loss, and epochs without improvement, if provided
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        training_state_path = os.path.join(
            checkpoint_folder_path, "training_state.json"
        )
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            return (
                training_state.get("epoch", 0),
                training_state.get("best_val_loss", float("inf")),
                training_state.get("epochs_without_improvement", 0),
            )
    return 0, float("inf"), 0
