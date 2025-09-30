from typing import Optional

from src.architectures.baseline_t5.target_generator.label_manipulation import (
    replace_label_with_full_name,
    remove_reserved_chars,
)


def dfs(
    parent_child_map: dict[str, list[str]],
    start_node,
    level_string_lists: list[list[str]],
    label2desc_file_path: str = None,
    do_replace_label_with_full_name: Optional[bool] = True,
) -> str:
    """
    Given the data in the different levels and the parent-child map, generate all the possible target sequences using
    a depth-first search (DFS) algorithm and concatenates them into a single string. The "Pop" keyword is used to
    indicate that the algorithm should go back to the parent node.

    :param parent_child_map: A map containing the parent-child relationships.
    :param start_node: The node from which the DFS algorithm should start.
    :param level_string_lists: A list in which every list contains the labels for a specific level in increasing order.
    :param label2desc_file_path: Path to the file containing the mapping from labels to extended descriptions.
    :param do_replace_label_with_full_name: Whether to replace the labels with their full names.

    :return: A string containing all the possible identified paths in the hierarchy, by using a depth-first search.
    """
    # If the labels must be replaced with their full names, do so
    if do_replace_label_with_full_name:
        for i, level_string_list in enumerate(level_string_lists):
            level_string_lists[i] = replace_label_with_full_name(
                level_string_list, label2desc_file_path=label2desc_file_path
            )

    possible_labels: list[str] | set[str] = []
    # Clean all the reserved characters from the labels (i.e., spaces, hyphens, and slashes)
    for i, level_string_list in enumerate(level_string_lists):
        level_string_lists[i] = remove_reserved_chars(level_string_list)
        possible_labels.extend(level_string_lists[i])

    # Build the list of all the possible labels in the hierarchy
    possible_labels = set(possible_labels)
    possible_labels.add("root")
    visited = set()
    # Initialize the list that indicates the level of each element in the stack
    level_node_stack: list[int] = [0]
    # Initialize the stack with the start node
    stack: list[str] = [start_node]
    # Initialize the DFS list that will contain all the target sequences, with Pop keywords indicating that the current
    # branch is not explored anymore but alternative branches are still to be explored
    dfs_order: list[str] = []
    level = -1
    # While there are labels in the stack
    while stack:
        # Get the first label in the stack for the current level
        label: str = stack.pop()
        # Get the current level index
        current_level: int = level_node_stack.pop()
        # If the label has not been visited and is in the list of possible nodes
        if label not in visited and label in possible_labels:
            # If the previously considered level is higher than the current level, then add to the DFS order the
            # necessary number of "Pop" operations to go back to the current level
            for _ in range(level - current_level + 1):
                dfs_order.append("Pop")
            # Add the current label to the DFS order and to the visited set, so that it is not considered again if
            # it is encountered
            dfs_order.append(label)
            visited.add(label)
            # If the label has some children and the label is in the list of valid labels
            if label in parent_child_map and label in possible_labels:
                # Add all the children of the current label to the stack in reverse order to maintain the left-to-right
                children = [
                    child
                    for child in parent_child_map[label][::-1]
                    if child in possible_labels
                ]
                stack.extend(children)
                # For each of the children, add a level index to the level stack
                level_node_stack.extend([current_level + 1] * len(children))
            # If the label is the root
            elif label == "root":
                # Get the children in reverse order to maintain the left-to-right order and add them to the stack
                children = level_string_lists[0][::-1]
                stack.extend(children)
                # For each of the children, add a level index to the level stack
                level_node_stack.extend([current_level + 1] * len(children))
        # Update the level index with the current level
        level = current_level

    # For each level up to the current one, add a Pop operation to go back to the root
    for i in range(level):
        dfs_order.append("Pop")

    # Join all the elements in the DFS order list with a space
    return " ".join(dfs_order)
