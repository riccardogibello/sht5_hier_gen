import os
import pandas as pd
from matplotlib import pyplot as plt, font_manager as fm
from matplotlib.pyplot import table
from tokenizers import Tokenizer
import torch


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    train_variances: list[float],
    val_variances: list[float],
    learning_rates: list[float],
    save_path=None,
):
    """
    Plot the training and validation losses along with learning rates in separate plots and save the plot to a file
    if a path is provided.

    :param train_losses: List of average training losses.
    :param val_losses: List of average validation losses.
    :param train_variances: List of training loss variances.
    :param val_variances: List of validation loss variances.
    :param learning_rates: List of learning rates.
    :param save_path: Path to save the plot image.
    """
    epochs = range(1, len(train_losses) + 1)
    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    min_val_loss = min(val_losses)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training and validation losses with variances
    ax1.plot(epochs, train_losses, "b", label="Training loss")
    ax1.fill_between(
        epochs,
        [train_losses[i] - train_variances[i] for i in range(len(train_losses))],
        [train_losses[i] + train_variances[i] for i in range(len(train_losses))],
        color="b",
        alpha=0.2,
    )
    ax1.plot(epochs, val_losses, "r", label="Validation loss")
    ax1.fill_between(
        epochs,
        [val_losses[i] - val_variances[i] for i in range(len(val_losses))],
        [val_losses[i] + val_variances[i] for i in range(len(val_losses))],
        color="r",
        alpha=0.2,
    )
    ax1.axvline(
        x=min_val_loss_epoch,
        color="k",
        linestyle="--",
        label=f"Best Val Loss: {min_val_loss:.4f}",
    )
    ax1.scatter(min_val_loss_epoch, min_val_loss, color="red", zorder=5)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot learning rates
    ax2.plot(epochs, learning_rates, "g", label="Learning rate")
    ax2.axvline(x=min_val_loss_epoch, color="k", linestyle="--")
    ax2.scatter(
        min_val_loss_epoch,
        learning_rates[min_val_loss_epoch - 1],
        color="red",
        zorder=5,
    )
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Learning Rate")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def store_model_metrics(
    _output_file_path: str,
    _model_metrics: dict[str, dict[str, float]],
):
    if len(_model_metrics) == 0:
        raise ValueError("No model metrics provided.")

    # Load existing metrics if the file exists, otherwise create an empty DataFrame
    try:
        df = pd.read_csv(_output_file_path)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["model_name"] + list(next(iter(_model_metrics.values())).keys())
        )

    # Convert the new metrics to a DataFrame
    new_metrics = pd.DataFrame.from_dict(
        _model_metrics,
        # Pass orient='index' so that the keys of the dictionary will be the rows
        orient="index",
    ).reset_index()
    new_metrics.columns = ["model_name"] + list(new_metrics.columns[1:])

    # Merge the new metrics with the existing ones, updating existing entries
    df = pd.concat([df, new_metrics], axis=0, ignore_index=True)
    # Keep only the last entry for each model, so that new metrics will overwrite old ones
    df = df.groupby("model_name", as_index=False).last()

    # Save the updated DataFrame back to the file
    df.to_csv(_output_file_path, index=False)


def store_per_label_metrics(
    _output_file_path: str,
    _per_label_model_metrics: dict[str, dict[str, dict[str, float]]],
):
    # Get a random dictionary from the per-label model metrics
    tmp_dict = list(list(_per_label_model_metrics.values())[0].values())[0]
    # Extract from one of the per-label dictionaries the metrics
    metric_names = list(tmp_dict.keys())

    initial_columns = ["model_name", "label"]
    initial_columns.extend(metric_names)
    try:
        # Open, if already existing, the file containing the per-label metrics
        df = pd.read_csv(_output_file_path)
        # Remove any column that is not in the specified initial columns, and add the missing ones
        for column in df.columns:
            if column not in initial_columns:
                df = df.drop(column, axis=1)
        for column in initial_columns:
            if column not in df.columns:
                df[column] = None
    except FileNotFoundError:
        df = pd.DataFrame(columns=initial_columns)

    # For each model name and label, update the metrics if already existing, otherwise add them
    for model_name, label_metrics in _per_label_model_metrics.items():
        for label, metrics in label_metrics.items():
            # Check if the model name and label are already in the DataFrame
            if pd.Series(
                (df["model_name"] == model_name) & (df["label"] == label)
            ).any():
                # Update the metrics
                df.loc[
                    (df["model_name"] == model_name) & (df["label"] == label),
                    metric_names,
                ] = metrics.values()
            else:
                total_samples = metrics.pop("total_samples")
                new_row = pd.DataFrame(
                    [
                        {
                            "model_name": model_name,
                            "label": label,
                            "total_samples": total_samples,
                            **metrics,
                        }
                    ]
                )
                df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the file
    df.to_csv(_output_file_path, index=False)


def store_mappings(
    file_folder: str,
    file_name: str,
    src_texts: torch.Tensor,
    true_label_tensor: torch.Tensor,
    predicted_logits_or_indices: torch.FloatTensor,
    input_tokenizer: Tokenizer,
    output_tokenizer: Tokenizer,
) -> None:
    """
    This method creates, or appends to, a CSV file containing the mappings between the source texts, the true labels,
    and the predicted labels. The labels are space-separated strings containing the hierarchical labels.
    """
    import csv

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    file_path = os.path.join(file_folder, file_name)
    header = ["source_string", "true_string", "predicted_string"]

    if len(predicted_logits_or_indices.shape) == 3:
        # Translate the predicted logits into the corresponding token identifiers
        pred_indices = torch.argmax(predicted_logits_or_indices, dim=-1)
    else:
        pred_indices = predicted_logits_or_indices

    # Move both tensors to the CPU and convert them to numpy arrays
    true_label_tensor = true_label_tensor.to("cpu").numpy()
    pred_indices = pred_indices.to("cpu").numpy()
    # Replace the -100 values with 0
    true_label_tensor[true_label_tensor == -100] = 0

    # Prepare new rows
    new_rows = zip(
        input_tokenizer.batch_decode(src_texts, skip_special_tokens=True),
        output_tokenizer.batch_decode(true_label_tensor, skip_special_tokens=False),
        output_tokenizer.batch_decode(pred_indices, skip_special_tokens=False),
    )

    file_exists = os.path.exists(file_path)
    # If file does not exist, write header and all rows
    if not file_exists:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(new_rows)
    else:
        # Only append new rows, do not read the whole file
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
