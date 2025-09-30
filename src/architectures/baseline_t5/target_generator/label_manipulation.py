import pandas as pd

from src.architectures.baseline_t5.target_generator.target_sequence import (
    label2desc_reduced,
)


def replace_label_with_full_name(
    label_list: list[str], label2desc_file_path: str = None
) -> list[str]:
    """
    Replace labels with their full names using a provided mapping.

    :param label_list: The list of labels for which the description must be retrieved.
    :param label2desc_file_path: Path to the file containing the mapping from labels to extended descriptions.

    :return: The list of descriptions corresponding to the given labels.
    """

    def load_label2desc(_labels_csv_file_path: str) -> dict[str, str]:
        """

        :param _labels_csv_file_path:
        :return:
        """
        # Load the label dataframe
        label_df = pd.read_csv(_labels_csv_file_path)
        # Get the labels and their descriptions
        label2title: dict[str, str] = {}
        for label, description in zip(label_df["label"], label_df["description"]):
            label2title[label] = description

        return label2title

    # If the path to the file containing the label metadata was not given, use the default one
    if label2desc_file_path is None:
        _label2desc_reduced: dict[str, str] = label2desc_reduced
    else:
        _label2desc_reduced: dict[str, str] = load_label2desc(label2desc_file_path)

    return [_label2desc_reduced[label] for label in label_list]


def remove_reserved_chars_in_map(
    parent_children: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Remove reserved characters (spaces, hyphens, and slashes) from the keys and values in the parent-child map.

    :param parent_children: Dictionary with parent-child relationships.

    :return: Cleaned dictionary with reserved characters removed.
    """

    def remove_spaces_from_list(lst: list[str]) -> list[str]:
        """
        Helper function to remove reserved characters from a list of strings.

        :param lst: List of strings to clean.

        :return: List of cleaned strings.
        """
        return [
            string.replace(" ", "").replace("-", "").replace("/", "") for string in lst
        ]

    parent_children_clean = {}
    for key, value in parent_children.items():
        cleaned_key = key.replace(" ", "").replace("-", "").replace("/", "")
        cleaned_value = remove_spaces_from_list(value)
        parent_children_clean[cleaned_key] = cleaned_value

    return parent_children_clean


def remove_reserved_chars(
    l_strings: list[str],
) -> list[str]:
    """
    Remove reserved characters (spaces, hyphens, and slashes) from the labels in the provided lists.

    :param l_strings: List of level strings.

    :return: Tuple containing cleaned lists of labels for each level.
    """

    def clean_labels(labels: list[str]) -> list[str]:
        """
        Helper function to remove reserved characters from a list of labels.

        :param labels: List of labels to clean.

        :return: List of cleaned labels.
        """
        return [
            label.replace(" ", "").replace("-", "").replace("/", "") for label in labels
        ]

    return clean_labels(l_strings)
