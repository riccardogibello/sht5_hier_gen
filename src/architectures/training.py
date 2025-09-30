import csv
import os
import sys
import time
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.architectures.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from src.architectures.structured_generative_t5.config import (
    TrainingConfig,
)
from src.evaluation.output import plot_losses
from src.utils.constants import CHECKPOINT_FOLDER_NAME
from src.utils.learning_rate import compute_lr_factor

from torch.optim.lr_scheduler import LambdaLR

from src.utils.model_utils import set_mode


def _losses_file_path(model_folder_path: str) -> str:
    return os.path.join(model_folder_path, "losses_and_variances.csv")


def _load_losses_and_variances(model_folder_path: str):
    file_path = _losses_file_path(model_folder_path)
    train_losses, train_variances, val_losses, val_variances, l_rates = (
        [],
        [],
        [],
        [],
        [],
    )
    if os.path.exists(file_path):
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_losses.append(float(row["train_loss"]))
                train_variances.append(float(row["train_variance"]))
                val_losses.append(float(row["val_loss"]))
                val_variances.append(float(row["val_variance"]))
                l_rates.append(float(row["learning_rate"]))
    return train_losses, train_variances, val_losses, val_variances, l_rates


def _append_losses_and_variances(
    model_folder_path: str,
    epoch: int,
    train_loss: float,
    train_variance: float,
    val_loss: float,
    val_variance: float,
    learning_rate: float,
):
    file_path = _losses_file_path(model_folder_path)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_variance",
                    "val_loss",
                    "val_variance",
                    "learning_rate",
                ]
            )
        writer.writerow(
            [epoch, train_loss, train_variance, val_loss, val_variance, learning_rate]
        )


def train(
    model: nn.Module,
    training_config: TrainingConfig,
    train_dataloader: Any,
    model_folder_path: str,
    val_dataloader: Optional[Any] = None,
    store_mappings_callback: Optional[callable] = None,
) -> None:
    def after_warmup_callback():
        set_mode(
            model,
            parameter_names=training_config.after_warmup_parameters,
            do_train=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.adam_lr,
    )
    early_stopping_patience = training_config.early_stopping_patience
    accumulation_steps = training_config.accum_iter

    if training_config.warmup_steps is None:
        assert (
            training_config.warmup_epochs is not None
        ), "Warmup steps or epochs must be provided."
        warmup_steps = int(training_config.warmup_epochs * len(train_dataloader))
        training_config.warmup_steps = warmup_steps
        warmup_steps = warmup_steps // accumulation_steps
        warmup_epochs = training_config.warmup_epochs
    else:
        warmup_steps = training_config.warmup_steps
        warmup_epochs = training_config.warmup_steps // len(train_dataloader)

    do_linear_lr = training_config.do_linear_lr

    warmup_scheduler = (
        LambdaLR(
            optimizer,
            lr_lambda=lambda step: compute_lr_factor(
                step,
                warmup_steps=warmup_steps,
                max_steps=training_config.max_steps,
            ),
        )
        if do_linear_lr
        else None
    )

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
    )

    do_scale = training_config.do_scale
    scaler: Optional[torch.GradScaler] = (
        torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if do_scale
        else None
    )

    start_epoch, best_val_loss, epochs_without_improvement = load_checkpoint(
        model,
        model_folder_path,
        optimizer,
        warmup_scheduler,
    )

    train_losses, train_variances, val_losses, val_variances, l_rates = (
        _load_losses_and_variances(
            model_folder_path,
        )
    )

    lr_reset_done = False

    for epoch in range(start_epoch, training_config.max_epochs):
        set_mode(
            model,
            parameter_names=(
                training_config.warmup_parameters
                if epoch < warmup_epochs
                else training_config.after_warmup_parameters
            ),
            do_train=True,
        )

        # Set the learning rate to the after warmup value
        if epoch >= warmup_epochs and not lr_reset_done:
            for param_group in optimizer.param_groups:
                param_group["lr"] = training_config.after_warmup_adam_lr
            lr_reset_done = True

        scheduler_to_use = warmup_scheduler if epoch < warmup_epochs else None

        avg_train_loss, avg_train_variance = model.step(
            data_generator=train_dataloader,
            log_message_prefix="train",
            optimizer=optimizer,
            scheduler=scheduler_to_use,
            previous_steps=len(train_dataloader) * epoch // accumulation_steps,
            training_config=training_config,
            store_mappings_callback=store_mappings_callback,
            after_warmup_callback=after_warmup_callback,
            scaler=scaler,
        )
        train_losses.append(avg_train_loss)
        train_variances.append(avg_train_variance)
        l_rates.append(optimizer.param_groups[0]["lr"])

        if val_dataloader is not None:
            with torch.no_grad():
                set_mode(model, do_train=False)
                avg_val_loss, avg_val_variance = model.step(
                    val_dataloader,
                    log_message_prefix="eval",
                    store_mappings_callback=store_mappings_callback,
                    training_config=training_config,
                )
                val_losses.append(avg_val_loss)
                val_variances.append(avg_val_variance)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    save_checkpoint(
                        model,
                        os.path.join(model_folder_path, CHECKPOINT_FOLDER_NAME, "best"),
                        epoch + 1,
                        best_val_loss,
                        epochs_without_improvement,
                        optimizer,
                        warmup_scheduler,
                    )
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break

                # After warmup, update ReduceLROnPlateau
                if epoch >= warmup_epochs:
                    plateau_scheduler.step(avg_val_loss)
        else:
            val_losses.append(0)
            val_variances.append(0)

        # Store losses and variances for this epoch
        _append_losses_and_variances(
            model_folder_path,
            epoch + 1,
            avg_train_loss,
            avg_train_variance,
            val_losses[-1],
            val_variances[-1],
            l_rates[-1],
        )

        save_checkpoint(
            model,
            os.path.join(model_folder_path, CHECKPOINT_FOLDER_NAME),
            epoch + 1,
            best_val_loss,
            epochs_without_improvement,
            optimizer,
            warmup_scheduler,
        )

        plot_losses(
            train_losses,
            val_losses,
            train_variances,
            val_variances,
            l_rates,
            os.path.join(model_folder_path, "metrics.png"),
        )

    assert (
        len(train_losses)
        == len(val_losses)
        == len(l_rates)
        == len(train_variances)
        == len(val_variances)
    )


def test(
    model: nn.Module,
    test_dataloader: Any,
    model_folder_path: str,
    store_mappings_callback: callable,
    training_config: TrainingConfig,
    use_cache: Optional[bool] = True,
):
    if not os.path.exists(model_folder_path):
        raise FileNotFoundError(
            f"Model checkpoint folder not found in {model_folder_path}"
        )

    # Load the last best checkpoint of the model
    load_checkpoint(
        model=model,
        model_folder_path=model_folder_path,
        load_best=True,
    )
    # Set the model in evaluation mode
    set_mode(
        model,
        do_train=False,
    )

    with torch.no_grad():
        model.step(
            data_generator=test_dataloader,
            log_message_prefix="test",
            use_cache=use_cache,
            store_mappings_callback=store_mappings_callback,
            training_config=training_config,
        )
