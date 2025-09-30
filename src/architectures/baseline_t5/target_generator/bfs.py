from typing import Optional

from src.architectures.baseline_t5.target_generator.label_manipulation import (
    replace_label_with_full_name,
    remove_reserved_chars,
)


def bfs_combined(
    level_string_lists: list[list[str]],
    bottom_top: bool = False,
    label2desc_file_path: str = None,
    do_replace_label_with_full_name: Optional[bool] = True,
) -> str:
    """
    Build a path string from the provided label lists, either in BFS or bottom-top BFS order.

    :param level_string_lists: A list in which every list contains the labels for a specific level in increasing order.
    :param bottom_top: Boolean indicating if the path should be built in bottom-top order.
    :param label2desc_file_path: Path to the file containing the mapping from labels to extended descriptions.
    :param do_replace_label_with_full_name: Whether to replace the labels with their full names.

    :return: The constructed path string.
    """
    # If the labels must be replaced with the label descriptions, do so
    if do_replace_label_with_full_name:
        for i, level_string_list in enumerate(level_string_lists):
            level_string_lists[i] = replace_label_with_full_name(
                level_string_list, label2desc_file_path=label2desc_file_path
            )

    # Clean all the reserved characters from the labels (i.e., spaces, hyphens, and slashes)
    for i, level_string_list in enumerate(level_string_lists):
        level_string_lists[i] = remove_reserved_chars(level_string_list)

    # Invert the order of the levels if the bottom-top order is requested
    if bottom_top:
        level_string_lists = level_string_lists[::-1]

    # Build the path string from the provided labels
    path = ""
    for level_strings in level_string_lists:
        level_strings: list[str]
        # For each label in the current level
        for i in range(len(level_strings)):
            # If it is the first label in the level
            if i == 0:
                # If a previous label is already added to the path, add a / separator to indicate a parent-child
                # relationship
                if path != "":
                    path += "/ " + level_strings[i]
                else:
                    path += level_strings[i]
            else:
                # If it is not the first label in the level, add a - separator to indicate a sibling relationship
                path += "- " + level_strings[i]

    return path
