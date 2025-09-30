import os
import sys
import pandas as pd

from datasets import load_dataset
import torch

from torch.utils.data import DataLoader

from src.data_preprocessing.label_handling import format_labels
from src.utils.constants import (
    DEBUG,
    INPUT_FOLDER_NAME,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
    VALIDATION_FILE_NAME,
)
from src.utils.model_utils import ModelName


def extract_label_weights(
    complete_dataset,
    output_tokenizer,
) -> torch.Tensor:
    # Add original_label column to the dataset by reversing the tokenization
    def reverse_tokenization(example):
        # Get the tokenized labels
        original_labels = []

        for label_ids in example["labels"]:
            # Decode the tokenized labels to get the original labels
            decoded_labels = output_tokenizer.decode(
                label_ids, skip_special_tokens=True
            )
            # Append the decoded labels to the list
            original_labels.append(decoded_labels)
        # Add the original labels to the example
        example["original_labels"] = original_labels
        return example

    complete_dataset = complete_dataset.map(reverse_tokenization, batched=True)

    all_labels = []
    for split in complete_dataset.keys():
        all_labels.extend(complete_dataset[split]["original_labels"])

    label_counts = {}
    for label in all_labels:
        current_labels = label.split(" ")
        for current_label in current_labels:
            if current_label not in label_counts:
                label_counts[current_label] = 1
            else:
                label_counts[current_label] += 1

    token_weights = {}
    total = sum(label_counts.values())

    for token, count in label_counts.items():
        token_weights[token] = total / count  # inverse frequency

    # Compute inverse frequency weights
    token_weights = {token: total / count for token, count in label_counts.items()}

    # Normalize so that average weight is 1 (not max = 1)
    mean_weight = sum(token_weights.values()) / len(token_weights)
    token_weights = {
        token: weight / mean_weight for token, weight in token_weights.items()
    }

    # Convert to tensor based on vocab indices
    weights = torch.ones(len(output_tokenizer.get_vocab()))
    for token, weight in token_weights.items():
        idx = output_tokenizer.convert_tokens_to_ids(token)
        weights[idx] = weight

    return weights


from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import torch
import os
import pandas as pd


"""def collate_fn(
    batch_tok_ids: list[dict[str, torch.Tensor]],
    max_input_length: int = None,
    max_label_length: int = None,
    pad_token_id: int = 0,
):
    if max_input_length is None:
        max_input_length = max(len(x["input_ids"]) for x in batch_tok_ids)
    if max_label_length is None:
        max_label_length = max(len(x["labels"]) for x in batch_tok_ids)

    input_ids = torch.tensor(
        [
            x["input_ids"] + [pad_token_id] * (max_input_length - len(x["input_ids"]))
            for x in batch_tok_ids
        ]
    )
    labels = torch.tensor(
        [
            x["labels"] + [pad_token_id] * (max_label_length - len(x["labels"]))
            for x in batch_tok_ids
        ]
    )
    token_type_ids = torch.tensor(
        [
            x.get("token_type_ids", [pad_token_id] * len(x["input_ids"]))
            + [pad_token_id] * (max_input_length - len(x["input_ids"]))
            for x in batch_tok_ids
        ]
    )
    attention_mask = (input_ids != pad_token_id).float()

    return input_ids, attention_mask, labels, token_type_ids


def bucket_and_batch_dataset(dataset, batch_size, seed=42, bucket_size=1000):
    dataset = dataset.shuffle(seed=seed)
    buckets = []
    for i in range(0, len(dataset), bucket_size):
        chunk = dataset.select(range(i, min(i + bucket_size, len(dataset))))
        sorted_chunk = chunk.sort("input_length")
        buckets.append(sorted_chunk)
    return concatenate_datasets(buckets)


def load_data(
    input_tokenizer,
    output_tokenizer,
    max_output_length: int,
    experiment_data_folder: str,
    output_file_name: str,
    first_level_chars: int,
    per_level_chars: int,
    max_level: int,
    batch_size: int,
    model_name,
    cumulative: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    pt_dataset_file_path = os.path.join(
        experiment_data_folder,
        "input",
        output_file_name,
    )

    if not os.path.exists(pt_dataset_file_path):
        data_files = {
            name.split(".")[0]: os.path.join(experiment_data_folder, "input", name)
            for name in ["train.csv", "validation.csv", "test.csv"]
        }
        for path in data_files.values():
            assert os.path.exists(path), f"File {path} does not exist."

        complete_dataset = load_dataset(
            "csv",
            data_files=data_files,
            delimiter=",",
        )

        complete_dataset = complete_dataset.remove_columns(["category"])
        complete_dataset = complete_dataset.filter(
            lambda x: x["description"] is not None and pd.notna(x["description"]),
            load_from_cache_file=False,
        )
        complete_dataset = complete_dataset.filter(
            lambda x: x["label"] is not None and pd.notna(x["label"]),
            load_from_cache_file=False,
        )

        complete_dataset = complete_dataset.map(
            format_labels,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={
                "first_level_chars": first_level_chars,
                "per_level_chars": per_level_chars,
                "max_level": max_level,
                "cumulative": cumulative,
                "model_name": model_name,
            },
        )

        complete_dataset = complete_dataset.filter(
            lambda x: len(x["label"]) > 0 and pd.notna(x["label"]),
            load_from_cache_file=False,
        )

        def tokenize_batch(batch):
            input_encodings = input_tokenizer(
                batch["description"],
                max_length=input_tokenizer.model_max_length - 1,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            label_encodings = output_tokenizer(
                batch["label"],
                max_length=max_output_length - 1,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            token_type_ids = input_encodings.get("token_type_ids", None)
            return {
                "input_ids": input_encodings["input_ids"],
                "labels": label_encodings["input_ids"],
                "input_length": [len(seq) for seq in input_encodings["input_ids"]],
                **(
                    {"token_type_ids": token_type_ids}
                    if token_type_ids is not None
                    else {}
                ),
            }

        complete_dataset = complete_dataset.map(
            tokenize_batch,
            batched=True,
            load_from_cache_file=False,
            batch_size=500,
        )

        torch.save(
            complete_dataset,
            pt_dataset_file_path,
        )
    else:
        complete_dataset = torch.load(pt_dataset_file_path, weights_only=False)

    label_weights = extract_label_weights(complete_dataset, output_tokenizer)

    # Apply bucketing and batching to train and val
    bucketed_train = bucket_and_batch_dataset(complete_dataset["train"], batch_size)
    bucketed_val = bucket_and_batch_dataset(complete_dataset["validation"], batch_size)

    train_dataloader = DataLoader(
        bucketed_train,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda input: collate_fn(
            input,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )
    val_dataloader = DataLoader(
        bucketed_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda input: collate_fn(
            input,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )
    test_dataloader = DataLoader(
        complete_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda input: collate_fn(
            input,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )

    return train_dataloader, val_dataloader, test_dataloader, label_weights"""


def collate_fn(
    batch_tok_ids: list[dict[str, torch.Tensor]],
    max_input_length: int,
    max_label_length: int,
    pad_token_id: int = 0,
):
    # Pad the input IDs and labels to the maximum lengths
    input_ids = torch.tensor(
        [
            x["input_ids"] + [pad_token_id] * (max_input_length - len(x["input_ids"]))
            for x in batch_tok_ids
        ]
    )
    labels = torch.tensor(
        [
            x["labels"] + [pad_token_id] * (max_label_length - len(x["labels"]))
            for x in batch_tok_ids
        ]
    )
    token_type_ids = torch.tensor(
        [
            (
                x["token_type_ids"]
                + [pad_token_id] * (max_input_length - len(x["token_type_ids"]))
                if "token_type_ids" in x and x["token_type_ids"] is not None
                else [pad_token_id] * max_input_length
            )
            for x in batch_tok_ids
        ]
    )

    # Create an attention mask for the input IDs
    attention_mask = (input_ids != pad_token_id).float()

    return input_ids, attention_mask, labels, token_type_ids


def load_data(
    input_tokenizer,
    output_tokenizer,
    max_output_length: int,
    experiment_data_folder: str,
    output_file_name: str,
    first_level_chars: int,
    per_level_chars: int,
    max_level: int,
    batch_size: int,
    model_name: ModelName,
    cumulative: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    pt_dataset_file_path = os.path.join(
        experiment_data_folder,
        INPUT_FOLDER_NAME,
        output_file_name,
    )
    encoder_max_length = 256  # input_tokenizer.model_max_length

    if not os.path.exists(pt_dataset_file_path):
        data_files = {
            name.split(".")[0]: os.path.join(
                experiment_data_folder,
                INPUT_FOLDER_NAME,
                name,
            )
            for name in [TRAIN_FILE_NAME, VALIDATION_FILE_NAME, TEST_FILE_NAME]
        }
        for path in data_files.values():
            assert os.path.exists(path), f"File {path} does not exist."

        complete_dataset = load_dataset(
            "csv",
            data_files=data_files,
            delimiter=",",
        )
        # Remove the 'category' column if it exists
        if "category" in complete_dataset.column_names["train"]:
            complete_dataset = complete_dataset.remove_columns(["category"])
        complete_dataset = complete_dataset.filter(
            lambda x: x["description"] is not None and pd.notna(x["description"]),
            load_from_cache_file=False,
        )
        complete_dataset = complete_dataset.filter(
            lambda x: x["label"] is not None and pd.notna(x["label"]),
            load_from_cache_file=False,
        )
        complete_dataset = complete_dataset.map(
            format_labels,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={
                "first_level_chars": first_level_chars,
                "per_level_chars": per_level_chars,
                "max_level": max_level,
                "cumulative": cumulative,
                "model_name": model_name,
            },
        )
        # Drop any row in which the label is empty
        complete_dataset = complete_dataset.filter(
            lambda x: len(x["label"]) > 0 and pd.notna(x["label"]),
            load_from_cache_file=False,
        )

        def tokenize_batch(batch):
            # Tokenize input descriptions
            input_encodings = input_tokenizer(
                batch["description"],
                max_length=encoder_max_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            # Tokenize output labels
            label_encodings = output_tokenizer(
                batch["label"],
                max_length=max_output_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            token_type_ids = input_encodings.get("token_type_ids", None)
            returned_map = {
                "input_ids": input_encodings["input_ids"],
                "labels": label_encodings["input_ids"],
            }
            if token_type_ids is not None:
                returned_map["token_type_ids"] = token_type_ids
            return returned_map

        complete_dataset = complete_dataset.map(
            tokenize_batch,
            batched=True,
            load_from_cache_file=False,
            batch_size=500,
        )

        # Cache the dataset in memory to avoid tokenizing it again
        torch.save(
            complete_dataset,
            os.path.join(
                experiment_data_folder,
                INPUT_FOLDER_NAME,
                output_file_name,
            ),
        )
    else:
        # Load the cached dataset
        complete_dataset = torch.load(
            pt_dataset_file_path,
            weights_only=False,
        )

    # Clip any of the dataset's splits to 100 samples for debugging purposes
    if DEBUG:
        for split in complete_dataset.keys():
            complete_dataset[split] = (
                complete_dataset[split].shuffle(seed=42).select(range(100))
            )

    label_weights = extract_label_weights(
        complete_dataset,
        output_tokenizer,
    )

    # Build the data loaders for the training, validation, and test sets
    train_dataloader = DataLoader(
        complete_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda input: collate_fn(
            input,
            max_input_length=encoder_max_length,
            max_label_length=max_output_length,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )
    val_dataloader = DataLoader(
        complete_dataset["validation"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda input: collate_fn(
            input,
            max_input_length=encoder_max_length,
            max_label_length=max_output_length,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )
    test_dataloader = DataLoader(
        complete_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda input: collate_fn(
            input,
            max_input_length=encoder_max_length,
            max_label_length=max_output_length,
            pad_token_id=input_tokenizer.pad_token_id,
        ),
    )

    return train_dataloader, val_dataloader, test_dataloader, label_weights
