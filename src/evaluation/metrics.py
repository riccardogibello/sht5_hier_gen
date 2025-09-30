import os.path
from typing import List, Any

import numpy as np
from sklearn.metrics import f1_score

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.data_preprocessing.label_handling import (
    build_label_tree,
    extract_label_components,
    extract_tree_labels,
)
from src.evaluation.hemkit_interface import build_hemkit_files, run_hemkit_executable
from src.evaluation.output import store_model_metrics
from src.utils.constants import (
    METRICS_FILE_NAME,
    TEST_CLEANED_FILE_NAME,
    TEST_FILE_NAME,
)
from src.utils.model_utils import ModelName
from src.utils.os import find_files_with_substring


def _compute_traditional_classification_metrics(
    _true_labels: pd.Series,
    _predicted_labels: pd.Series,
):
    # Compute micro and macro F1 scores
    _micro_f1 = f1_score(
        _true_labels,
        _predicted_labels,
        average="micro",
    )
    _macro_f1 = f1_score(
        _true_labels,
        _predicted_labels,
        average="macro",
    )

    return {
        "micro_f1": _micro_f1,
        "macro_f1": _macro_f1,
    }


def _compute_hierarchical_classification_metrics(
    cleaned_test_file_path: str,
    base_data_folder_path: str,
    unique_labels: List[str],
    per_level_chars: int = 1000,
    first_level_chars: int = 2,
    maxdist: int = 1000000,
    maxerr: int = 5,
):
    exe_file_path, hierarchy_file, truecat_file, predcat_file = build_hemkit_files(
        base_data_folder_path=base_data_folder_path,
        cleaned_test_file_path=cleaned_test_file_path,
        unique_labels=unique_labels,
        first_level_chars=first_level_chars,
        per_level_chars=per_level_chars,
    )

    # Run the HEMKit executable with the provided parameters
    return run_hemkit_executable(
        hemkit_executable_path=exe_file_path,
        hierarchy_file=hierarchy_file,
        truecat_file=truecat_file,
        predcat_file=predcat_file,
        maxdist=maxdist,
        maxerr=maxerr,
    )


def compute_metrics(
    models_folder_path: str,
    labels_file_path: str,
    base_data_folder_path: str,
    first_level_chars: int,
    per_level_chars: int,
    model_name: ModelName,
) -> None:
    # Find all test files in the models folder path
    test_file_path_list = find_files_with_substring(
        base_path=models_folder_path,
        substring=TEST_FILE_NAME,
    )

    if len(test_file_path_list) == 0:
        return None
    else:
        # Load the file containing the labels that are valid and instantiate a LabelEncoder
        labels_df = pd.read_csv(labels_file_path)
        labels_df = labels_df.dropna(subset=["label"])
        # Get the unique label column
        unique_labels = list(labels_df["label"].unique())
        unique_labels.append("ERROR")
        label_encoder = MultiLabelBinarizer()
        label_encoder.fit([unique_labels])

        for level in range(1, 8):
            final_metrics_file_path = os.path.join(
                models_folder_path,
                METRICS_FILE_NAME.split(".")[0] + f"_{level}.csv",
            )

            metrics: dict[str, dict[str, Any]] = {}
            for test_file_path in test_file_path_list:
                cumulative = True if "cum_True" in test_file_path else False
                label_tree = build_label_tree(
                    labels=unique_labels,
                    max_level=1000,
                    first_level_chars=first_level_chars,
                    per_level_chars=per_level_chars,
                    cumulative=cumulative,
                    model_name=model_name,
                )
                directory_path = os.path.dirname(test_file_path)
                cleaned_test_file_path = os.path.join(
                    directory_path,
                    TEST_CLEANED_FILE_NAME,
                )

                # Extract the last but one component of the path, which contains the model name
                model_name = test_file_path.split(os.path.sep)[-3]

                # Load the CSV file containing the test data
                test_data = pd.read_csv(test_file_path)
                # Replace any NaN values with empty strings
                test_data = test_data.fillna("")
                source_strings = test_data["source_string"].tolist()
                true_labels = test_data["true_string"].tolist()
                predicted_labels = test_data["predicted_string"].tolist()
                # Remove from the predicted strings any "<pad> " substring
                predicted_labels = [
                    predicted_label.replace("<pad> ", "")
                    for predicted_label in predicted_labels
                ]

                true_label_list: list = []
                predicted_label_list: list = []
                total_errors = 0
                total_true_labels = 0
                cleaned_true_labels = []
                cleaned_predicted_labels = []

                def refactor_label(label: str) -> str:
                    returned_label = "ROOT "
                    labels = label.split("$")
                    for i, label in enumerate(labels):
                        components = extract_label_components(
                            label=label,
                            first_level_chars=first_level_chars,
                            per_level_chars=per_level_chars,
                            model_name=model_name,
                            cumulative=cumulative,
                        )
                        if len(components) > 0:
                            returned_label += " ".join(components)
                            returned_label += " POP " * len(components)
                    returned_label += " </s>"
                    return returned_label

                for true_label, predicted_label in zip(true_labels, predicted_labels):
                    if "$" not in true_label or "$" not in predicted_label:
                        true_label = refactor_label(true_label)
                        predicted_label = refactor_label(predicted_label)
                    tmp_true_labels, _ = extract_tree_labels(
                        tree_formatted_string=true_label,
                        max_label_chars=1 + (level - 1) * per_level_chars,
                        first_level_chars=first_level_chars,
                        per_level_chars=per_level_chars,
                        label_tree=label_tree,
                        cumulative=cumulative,
                        model_name=ModelName.STRUCTURED_GENERATIVE_T5,  # TODO
                    )
                    tmp_true_labels = sorted(list(set(tmp_true_labels)))
                    cleaned_true_labels.append("$".join(tmp_true_labels))
                    true_label_list.append(
                        label_encoder.transform([tmp_true_labels]).flatten(),
                    )
                    tmp_predicted_labels, errors = extract_tree_labels(
                        tree_formatted_string=predicted_label,
                        max_label_chars=1 + (level - 1) * per_level_chars,
                        first_level_chars=first_level_chars,
                        per_level_chars=per_level_chars,
                        label_tree=label_tree,
                        cumulative=cumulative,
                        model_name=ModelName.STRUCTURED_GENERATIVE_T5,  # TODO
                    )
                    tmp_predicted_labels = sorted(list(set(tmp_predicted_labels)))
                    cleaned_predicted_labels.append("$".join(tmp_predicted_labels))
                    predicted_label_list.append(
                        label_encoder.transform([tmp_predicted_labels]).flatten(),
                    )
                    total_errors += errors
                    total_true_labels += len(tmp_true_labels)

                # Save the cleaned test data to a CSV file
                cleaned_test_df = pd.DataFrame(
                    {
                        "source_string": source_strings,
                        "true_string": cleaned_true_labels,
                        "predicted_string": cleaned_predicted_labels,
                    }
                )
                cleaned_test_df.to_csv(
                    cleaned_test_file_path,
                    index=False,
                )
                print(f"Cleaned test data saved to {cleaned_test_file_path}")

                per_label_error = (
                    total_errors / total_true_labels if total_true_labels > 0 else 0
                )
                print(
                    f"Total errors: {total_errors}, Total true labels: {total_true_labels}"
                )

                predicted_label_list = np.array(predicted_label_list)
                true_label_list = np.array(true_label_list)

                model_metrics = _compute_traditional_classification_metrics(
                    _true_labels=true_label_list,
                    _predicted_labels=predicted_label_list,
                )
                metrics[model_name] = model_metrics
                model_metrics = _compute_hierarchical_classification_metrics(
                    base_data_folder_path=base_data_folder_path,
                    cleaned_test_file_path=cleaned_test_file_path,
                    unique_labels=unique_labels,
                    per_level_chars=per_level_chars,
                    first_level_chars=first_level_chars,
                    maxdist=1000000,
                    maxerr=5,
                )
                metrics[model_name].update(model_metrics)
                metrics[model_name]["avg_error"] = per_label_error

            store_model_metrics(
                _model_metrics=metrics,
                _output_file_path=final_metrics_file_path,
            )
