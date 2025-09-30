import re
from typing import Any
import pandas as pd

from src.utils.model_utils import ModelName


def extract_label_components(
    label: str,
    first_level_chars: int,
    per_level_chars: int,
    model_name: ModelName,
    cumulative: bool = False,
) -> list[str]:
    # Extract from the label the first level characters (the last included)
    first_level_component = label[:first_level_chars]
    label = label[first_level_chars:]

    maximum_index = len(label)
    if maximum_index % per_level_chars != 0:
        maximum_index = (
            maximum_index + per_level_chars - (maximum_index % per_level_chars)
        )
    components = [first_level_component]
    if not cumulative:
        for i in range(0, maximum_index, per_level_chars):
            components.append(
                label[i : i + per_level_chars],
            )
    else:
        for i in range(0, maximum_index, per_level_chars):
            if not cumulative:
                components.append(
                    label[i : i + per_level_chars],
                )
            else:
                spacer = ""
                if model_name == ModelName.STRUCTURED_GENERATIVE_T5:
                    spacer = ""
                components.append(
                    components[-1] + spacer + label[i : i + per_level_chars],
                )
    return components


def build_label_tree(
    labels: list[str],
    max_level: int,
    first_level_chars: int,
    per_level_chars: int,
    model_name: ModelName,
    cumulative: bool = False,
) -> dict[str, Any]:
    """
    Build a hierarchical tree structure from the given labels.

    Args:
        labels (list[str]): List of labels to be processed. Each label is a $-separated string representing
        a hierarchical label.
        max_label_groups (int): The maximum hierarchical level of the labels to be kept.
        grouping_number (int): The number of characters in each label group.

    Returns:
        dict[str, Any]: A nested dictionary representing the hierarchical structure of the labels.
        The keys are the components of the labels, and the values are dictionaries representing the children
        of each component.
    """
    tree = {}
    for label in labels:
        label = label.split("$")[-1]
        components = extract_label_components(
            label=label,
            first_level_chars=first_level_chars,
            per_level_chars=per_level_chars,
            cumulative=cumulative,
            model_name=model_name,
        )
        if len(components) > max_level:
            continue
        node = tree
        for comp in components:
            node = node.setdefault(comp, {})
    return tree


def flatten_label_tree(
    tree: dict[str, Any],
) -> str:
    """
    Flatten the hierarchical tree structure into a string representation.

    Args:
        tree (dict[str, Any]): A nested dictionary representing the hierarchical structure of the labels.
        The keys are the components of the labels, and the values are dictionaries representing the children
        of each component.

    Returns:
        str: A string representation of the hierarchical structure, where each label component is separated by a space,
        and "POP" indicates the end of a branch, while "ROOT" indicates the root of the tree.
    """

    def dfs(
        node: dict[str, Any],
    ) -> list[str]:
        """
        Perform a depth-first search (DFS) on the tree and return a list of strings representing the
        hierarchical structure.

        Args:
            node (dict[str, Any]): The current node in the tree.

        Returns:
            list[str]: A list of strings representing the hierarchical structure.
        """
        result = []
        for k, v in node.items():
            result.append(k)
            result.extend(dfs(v))
            result.append("POP")
        return result

    return "ROOT " + " ".join(dfs(tree)).strip()


def extract_tree_labels(
    tree_formatted_string: str,
    max_label_chars: int,
    first_level_chars: int,
    per_level_chars: int,
    label_tree: dict[str, Any],
    model_name: ModelName,
    cumulative: bool = False,
) -> tuple[list[str], int]:
    """
    Extract labels from a string formatted as a hierarchical tree structure.

    Args:
        tree_formatted_string (str): A string representation of the hierarchical structure, where each label component
        is separated by a space, and "POP" indicates the end of a branch, while "ROOT" indicates the root of the tree.
        max_label_chars (int): The maximum number of characters in each label.
        grouping_number (int): The number of characters in each label group.
        label_tree (dict[str, Any]): The hierarchical tree structure.

    Returns:
        list[str]: A list of strings representing the extracted labels.
    """
    tree_formatted_string = tree_formatted_string.split("</s>")[0] + " </s>"
    # Replace multiple spaces with a single space
    tree_formatted_string = re.sub(r"\s+", " ", tree_formatted_string.strip())
    special_components = ["POP", "ROOT", "<s>", "</s>"]
    components = tree_formatted_string.split(" ")
    tmp_components = components.copy()
    # Remove all the special components from the list
    tmp_components = [comp for comp in tmp_components if comp not in special_components]
    all_unitary_lengths = all(len(comp) == 1 for comp in tmp_components)
    if all_unitary_lengths and cumulative:
        last_letter_position = 0
        last_character = None
        new_components = []
        # Iterate over the original components
        for i, comp in enumerate(components):
            if comp in special_components and comp != last_character:
                if comp == "POP":
                    # Get all the components from the last letter position to the current position
                    new_components.append("".join(components[last_letter_position:i]))
                elif comp == "</s>":
                    break
            # If the current component is a letter of the alphabet, store its position
            elif comp.isalpha():
                last_letter_position = i
            last_character = comp
        # Replace the original components with the new ones
        components = new_components
    current_components = []
    labels = []

    malformed_labels = 0

    while len(components) > 0:
        current_component = components.pop(0)
        is_special_component = any(
            current_component == special_component
            for special_component in special_components
        )
        if is_special_component:
            if current_component == "POP":
                if len(current_components) > 0:
                    current_component = current_components.pop()
            elif current_component == "ROOT":
                current_components.clear()
            elif current_component == "</s>":
                break
            elif current_component == "<s>":
                continue
            else:
                raise ValueError(
                    f"Unexpected component: {current_component}. Expected one of {special_components}."
                )
        else:
            previous_components = current_components.copy()
            if cumulative:
                current_components = [current_component]
                current_label = current_component
            else:
                current_components.append(current_component)
                current_label = "".join(current_components)

            after_first_level_length = len(current_label) - first_level_chars
            if after_first_level_length < 0:
                is_right_length = False
            else:
                is_right_length = after_first_level_length % per_level_chars == 0
            is_valid, last_valid_components = verify_label(
                label=current_label,
                first_level_chars=first_level_chars,
                per_level_chars=per_level_chars,
                tree=label_tree,
                cumulative=cumulative,
                model_name=model_name,
            )

            if len(current_label) > max_label_chars:
                current_components.clear()
            elif is_right_length and is_valid:
                if is_valid:
                    if all_unitary_lengths and cumulative:
                        tmp_components = extract_label_components(
                            label=current_label,
                            first_level_chars=first_level_chars,
                            per_level_chars=per_level_chars,
                            cumulative=cumulative,
                            model_name=model_name,
                        )
                        labels.extend(tmp_components)
                    else:
                        labels.append(current_label)
            elif is_right_length and not is_valid:
                malformed_labels += 1
                # TODO
                labels.append("ERROR")
                current_components = previous_components
            else:
                pass
    return labels, malformed_labels


def verify_label(
    label: str,
    first_level_chars: int,
    per_level_chars: int,
    tree: dict[str, Any],
    model_name: ModelName,
    cumulative: bool = False,
) -> tuple[bool, list[str]]:
    """
    Verify if a label is present in the tree structure.

    Args:
        label (str): The label to be verified.
        grouping_number (int): The number of characters in each hierarchical label group.
        tree (dict[str, Any]): The hierarchical tree structure.

    Returns:
        bool: True if the label is present in the tree, False otherwise.
    """
    components = extract_label_components(
        label=label,
        first_level_chars=first_level_chars,
        per_level_chars=per_level_chars,
        cumulative=cumulative,
        model_name=model_name,
    )
    node = tree
    last_valid_components = []
    for comp in components:
        if comp not in node:
            return False, last_valid_components
        last_valid_components.append(comp)
        node = node[comp]
    return True, last_valid_components


def format_labels(
    batch: dict[str, Any],
    max_level: int,
    first_level_chars: int,
    per_level_chars: int,
    model_name: ModelName,
    cumulative: bool = False,
) -> dict[str, Any]:
    """
    Format all the labels in the batch by adhering to the Depth First Search (DFS) algorithm.

    Args:
        batch (dict[str, Any]): The batch of data containing labels. Each label is formatted as a string of the
        form "label1&label2&...&labelN", where each "labelN" is a string representing a hierarchical label. Each hierarchical
        label is a string of the form "label1$label2$...$labelN", where each "labelN" is a string representing a component of the
        hierarchical label.
        max_label_groups (int): The maximum hierarchical level of the labels to be kept.
        grouping_number (int): The number of characters in each label group.
    """
    new_labels = []
    for label_str in batch["label"]:
        if label_str is None or not pd.notna(label_str):
            new_labels.append("")
            continue
        label_list = label_str.split("&")
        tree: dict[str, Any] = build_label_tree(
            labels=label_list,
            max_level=max_level,
            first_level_chars=first_level_chars,
            per_level_chars=per_level_chars,
            cumulative=cumulative,
            model_name=model_name,
        )
        if len(tree) == 0:
            continue
        new_labels.append(flatten_label_tree(tree))
    batch["label"] = new_labels
    return batch
