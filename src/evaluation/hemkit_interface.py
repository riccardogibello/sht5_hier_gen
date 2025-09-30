import os
import re
import subprocess
from typing import List

import numpy as np
import pandas as pd

from src.utils.constants import (
    GOLD_STANDARD_FILE_NAME,
    HEMKIT_FILE_NAME,
    HEMKIT_FOLDER_NAME,
    HIERARCHICAL_FILES_FOLDER_NAME,
    HIERARCHY_FILE_NAME,
    PREDICTED_FILE_NAME,
    SOFTWARE_FOLDER_NAME,
)


def _build_hierarchy_file(
    unique_labels: List[str],
    hierarchy_file_path: str,
    first_level_chars: int = 2,
    per_level_chars: int = 1000,
):
    def find_parent_label(
        label: str,
        first_level_chars: int,
        per_level_chars: int,
    ) -> str:
        """
        Find the parent label for a given label based on the label hierarchy.

        :param label: The current label.
        :param first_level_chars: Number of chars for the first level.
        :param per_level_chars: Number of chars per subsequent level.
        :return: The parent label if found, otherwise an empty string.
        """
        if len(label) == first_level_chars + per_level_chars:
            returned_label = label[:first_level_chars]
        elif len(label) > first_level_chars + per_level_chars:
            returned_label = label[:-per_level_chars]
        else:
            returned_label = None
        return returned_label

    # Sort the labels from the shortest to the longest
    unique_labels = sorted(unique_labels, key=len)
    # Remove the "ERROR" label if it exists
    if "ERROR" in unique_labels:
        unique_labels.remove("ERROR")
    label_ids = range(1, len(unique_labels) + 1)
    label_to_id = dict(zip(unique_labels, label_ids))

    # Create a hierarchy file with the label hierarchy
    with open(hierarchy_file_path, "w", encoding="utf-8") as hierarchy_file:
        for label in unique_labels:
            parent_label = find_parent_label(
                label,
                first_level_chars,
                per_level_chars,
            )
            if parent_label:
                hierarchy_file.write(
                    f"{label_to_id[parent_label]} {label_to_id[label]}\n"
                )
            else:
                hierarchy_file.write(f"0 {label_to_id[label]}\n")


def _build_gs_pred_files(
    cleaned_test_file_path: str,
    label_to_id: dict[str, int],
    gold_standard_file_path: str,
    predicted_file_path: str,
):
    def refactor_string(
        string: str,
        label_to_id: dict[str, int],
    ) -> str:
        """
        Refactor a string by replacing labels with their corresponding IDs.
        """
        strings = string.split(" ")
        joined_string = ""
        prev_string = ""
        for s in strings:
            if s in label_to_id:
                id = str(label_to_id[s])
                prev_string = s
            elif prev_string in label_to_id:
                id = str(label_to_id[prev_string])
            else:
                id = "0"
            joined_string += id + " "

        return joined_string.strip()

    test_df = pd.read_csv(cleaned_test_file_path)

    # Fill NaN values to avoid errors in string operations
    test_df["true_string"] = test_df["true_string"].fillna("")
    test_df["predicted_string"] = test_df["predicted_string"].fillna("")
    # Replace any "ERROR" label with an empty string
    test_df["true_string"] = test_df["true_string"].str.replace(
        "ERROR",
        "",
        regex=False,
    )
    test_df["predicted_string"] = test_df["predicted_string"].str.replace(
        "ERROR",
        "",
        regex=False,
    )

    test_df["true_string"] = test_df["true_string"].str.replace("$", " ")
    test_df["predicted_string"] = test_df["predicted_string"].str.replace("$", " ")

    # Replace multiple spaces with a single space
    test_df["true_string"] = test_df["true_string"].apply(
        lambda s: re.sub(r"\s+", " ", s.strip())
    )
    test_df["predicted_string"] = test_df["predicted_string"].apply(
        lambda s: re.sub(r"\s+", " ", s.strip())
    )

    test_df["true_string"] = test_df["true_string"].apply(
        lambda s: refactor_string(s, label_to_id)
    )
    test_df["predicted_string"] = test_df["predicted_string"].apply(
        lambda s: refactor_string(s, label_to_id)
    )
    true_strings = test_df["true_string"].tolist()
    predicted_string = test_df["predicted_string"].tolist()
    with open(gold_standard_file_path, "w") as gold_standard_file:
        for true_string in true_strings:
            if true_string == "":
                true_string = "0"
            gold_standard_file.write(f"{true_string}\n")
    with open(predicted_file_path, "w") as predicted_file:
        for predicted in predicted_string:
            if predicted == "":
                predicted = "0"
            predicted_file.write(f"{predicted}\n")


def build_hemkit_files(
    base_data_folder_path: str,
    cleaned_test_file_path: str,
    unique_labels: List[str],
    first_level_chars: int = 2,
    per_level_chars: int = 1000,
):
    hemkit_folder_path = os.path.join(
        base_data_folder_path,
        "..",
        HEMKIT_FOLDER_NAME,
    )
    exe_file_path = os.path.join(
        hemkit_folder_path,
        SOFTWARE_FOLDER_NAME,
        HEMKIT_FILE_NAME,
    )
    hierarchical_files_folder_path = os.path.join(
        hemkit_folder_path,
        HIERARCHICAL_FILES_FOLDER_NAME,
    )
    os.makedirs(hierarchical_files_folder_path, exist_ok=True)
    hierarchy_file_path = os.path.join(
        hierarchical_files_folder_path,
        HIERARCHY_FILE_NAME,
    )
    gold_standard_file_path = os.path.join(
        hierarchical_files_folder_path,
        GOLD_STANDARD_FILE_NAME,
    )
    predicted_file_path = os.path.join(
        hierarchical_files_folder_path,
        PREDICTED_FILE_NAME,
    )

    _build_hierarchy_file(
        unique_labels=unique_labels,
        hierarchy_file_path=hierarchy_file_path,
        per_level_chars=per_level_chars,
        first_level_chars=first_level_chars,
    )
    _build_gs_pred_files(
        cleaned_test_file_path=cleaned_test_file_path,
        label_to_id={label: i + 1 for i, label in enumerate(unique_labels)},
        gold_standard_file_path=gold_standard_file_path,
        predicted_file_path=predicted_file_path,
    )

    return (
        exe_file_path,
        hierarchy_file_path,
        gold_standard_file_path,
        predicted_file_path,
    )


def run_hemkit_executable(
    hemkit_executable_path: str,
    hierarchy_file: str,
    truecat_file: str,
    predcat_file: str,
    maxdist: int,
    maxerr: int,
) -> dict[str, str]:
    """
    Runs the HEMKit executable with the given parameters and returns its output as a string.

    Args:
        hemkit_executable_path: Path to the HEMKit executable (e.g., './bin/HEMKit').
        hierarchy_file: Path to the hierarchy file.
        truecat_file: Path to the true categories file.
        predcat_file: Path to the predicted categories file.
        maxdist: Maximum distance parameter (int).
        maxerr: Maximum error parameter (int).

    Returns:
        The standard output from the HEMKit executable as a string.
    """
    cmd = [
        hemkit_executable_path,
        hierarchy_file,
        truecat_file,
        predcat_file,
        str(maxdist),
        str(maxerr),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"HEMKit failed: {result.stderr}")

    metrics_dict = {}
    lines = result.stdout.splitlines()
    for line in lines:
        metrics_splits = line.split(" = ")
        if len(metrics_splits) == 2:
            metric_name = metrics_splits[0].strip().lower()
            # Replace any space with an underscore in the metric name
            metric_name = metric_name.replace(" ", "_")
            metric_value = metrics_splits[1].strip()
            metrics_dict[metric_name] = metric_value
    return metrics_dict
