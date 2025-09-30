from sklearn.metrics import f1_score


def split_list_by_given_sizes(
    label_list: list[int], per_level_sizes: list[int]
) -> list[list[int]]:
    """
    Split the given list of labels by the given sizes, which correspond to the hierarchy levels.

    :param label_list: The list of labels to split.
    :param per_level_sizes: The sizes of the hierarchy levels.

    :return: A list in which every element corresponds to the labels of a hierarchy level.
    """
    # Prepare the list that will contain the labels split by the given sizes, which correspond to the hierarchy levels
    split_list: list[list[int]] = []
    start = 0
    for size in per_level_sizes:
        # Get the labels of the current level and add them to the split list
        split_list.append(label_list[start : start + size])
        start += size
    return split_list


def micro_f1_global_and_per_level(
    targets: list[list[int]], predictions: list[list[int]], per_level_size: list[int]
):
    # Compute the micro F1 global
    micro_f1_global = f1_score(targets, predictions, average="micro", zero_division=0)
    # Divide both the target and prediction multi-hot vectors into a per-level list of multi-hot vectors
    targets_per_level: list[list[list[int]]] = [
        split_list_by_given_sizes(target, per_level_size) for target in targets
    ]
    predictions_per_level = [
        split_list_by_given_sizes(pred, per_level_size) for pred in predictions
    ]
    # Compute, for each hierarchy level, the micro F1 score
    micro_f1_per_level = [
        f1_score(
            [target_per_level[j] for target_per_level in targets_per_level],
            [pred_per_level[j] for pred_per_level in predictions_per_level],
            average="micro",
            zero_division=0,
        )
        for j in range(len(per_level_size))
    ]
    return {
        "micro_f1_global": micro_f1_global,
        "micro_f1_per_level": micro_f1_per_level,
    }


def macro_f1_global_and_per_level(
    targets: list[list[int]], predictions: list[list[int]], per_level_size: list[int]
) -> dict[str, float | list[float]]:
    """
    Compute the macro F1 score globally and per hierarchy level.

    :param targets: List of multi-hot vectors representing the true labels.
    :param predictions: List of multi-hot vectors representing the predicted labels.
    :param per_level_size: List of integers representing the number of labels at each hierarchy level.
    :return: Dictionary containing the global macro F1 score and the macro F1 score for each hierarchy level.
    """
    # Compute the macro F1 global
    macro_f1_global = f1_score(targets, predictions, average="macro", zero_division=0)
    # Divide both the target and prediction multi-hot vectors into a per-level list of multi-hot vectors
    targets_per_level: list[list[list[int]]] = [
        split_list_by_given_sizes(target, per_level_size) for target in targets
    ]
    predictions_per_level: list[list[list[int]]] = [
        split_list_by_given_sizes(pred, per_level_size) for pred in predictions
    ]
    # Compute, for each hierarchy level, the macro F1 score
    macro_f1_per_level: list[float] = [
        f1_score(
            [target_per_level[j] for target_per_level in targets_per_level],
            [pred_per_level[j] for pred_per_level in predictions_per_level],
            average="macro",
            zero_division=0,
        )
        for j in range(len(per_level_size))
    ]
    return {
        "macro_f1_global": macro_f1_global,
        "macro_f1_per_level": macro_f1_per_level,
    }
