from typing import List, Dict, Set, Optional

from src.architectures.baseline_t5.target_generator.label_manipulation import (
    replace_label_with_full_name,
    remove_reserved_chars,
)


def bottom_top_dfs(
    parent_child_map: Dict[str, List[str]],
    level_string_lists: list[list[str]],
    label2desc_file_path: str = None,
    do_replace_label_with_full_name: Optional[bool] = True,
) -> str:
    """
    Perform a bottom-top depth-first search (DFS) to generate a target sequence.

    :param parent_child_map: Dictionary mapping parent labels to child labels.
    :param level_string_lists: List of strings, divided by levels and increasing order.
    :param label2desc_file_path: Path to the file containing the mapping from labels to extended descriptions.
    :param do_replace_label_with_full_name: Whether to replace the labels with their full names.

    :return: The constructed target sequence.
    """

    def go_up(
        _all_labels: List[str],
        _child_parent_map: Dict[str, str],
        _explored: Set[str],
        _label: str,
        _parent_child_map: Dict[str, List[str]],
        _target: str,
    ) -> str:
        """
        Recursively go up the hierarchy to build the target sequence.

        :param _all_labels: List of all labels.
        :param _child_parent_map: Dictionary mapping child labels to parent labels.
        :param _explored: Set of explored labels.
        :param _label: Current label.
        :param _parent_child_map: Dictionary mapping parent labels to child labels.
        :param _target: Current target sequence.

        :return: Updated target sequence.
        """
        if _label in _child_parent_map:
            _parent = _child_parent_map[_label]
            if _parent != "root":
                _brothers = _parent_child_map[_parent]
                for _brother in _brothers:
                    if (
                        _brother != _label
                        and _brother in _all_labels
                        and _brother not in _explored
                    ):
                        _target += "- " + _brother
                        _explored.add(_brother)
                _target += "/ " + _parent
                _explored.add(_parent)
                _target = go_up(
                    _all_labels,
                    _child_parent_map,
                    _explored,
                    _parent,
                    _parent_child_map,
                    _target,
                )
        return _target

    # If the labels must be replaced with their full names, do so
    if do_replace_label_with_full_name:
        for i, level_string_list in enumerate(level_string_lists):
            level_string_lists[i] = replace_label_with_full_name(
                level_string_list, label2desc_file_path=label2desc_file_path
            )

    possible_labels: list[str] | set[str] = []
    # Reverse the order of the levels
    level_string_lists = level_string_lists[::-1]
    # Clean all the reserved characters from the labels (i.e., spaces, hyphens, and slashes)
    for i, level_string_list in enumerate(level_string_lists):
        level_string_lists[i] = remove_reserved_chars(level_string_list)
        possible_labels.extend(level_string_lists[i])

    # Create a map from child labels to parent labels
    child_parent_map = {v: k for k, value in parent_child_map.items() for v in value}

    explored = set()
    target = ""

    for label in possible_labels:
        if label in explored:
            continue
        if target == "":
            target += label
        else:
            target += " " + label
        explored.add(label)
        target = go_up(
            possible_labels, child_parent_map, explored, label, parent_child_map, target
        )

    return target
