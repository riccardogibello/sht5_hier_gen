from src.architectures.baseline_t5.metrics.f1 import (
    micro_f1_global_and_per_level,
    macro_f1_global_and_per_level,
)


def _constrained_output_with_respect_to_hierarchy(
    child_parent_indexes: dict[int, int], predicted_m_hot_vector: list[int]
) -> list[int]:
    """
    Clean the predicted multi-hot vector based on the hierarchy.

    :param child_parent_indexes: A map in which each key is the index of a child label and the value is the index of
    the parent label.
    :param predicted_m_hot_vector: The predicted multi-hot vector.

    :return: A cleaned multi-hot vector, in which labels which are predicted as positive but their parent is not
    predicted as positive are removed.
    """
    # Build a constrained multi-hot vector based on the hierarchy
    constrained_predicted_m_hot_vector = [0 for _ in range(len(predicted_m_hot_vector))]
    # For each possible label index
    for l_index in range(len(predicted_m_hot_vector)):
        # If the label is predicted as positive
        if predicted_m_hot_vector[l_index] >= 0.5:
            # If the label has a parent label
            if l_index in child_parent_indexes.keys():
                # If the parent label is also predicted as positive
                if predicted_m_hot_vector[child_parent_indexes[l_index]] >= 0.5:
                    # Then the label is also predicted as positive
                    constrained_predicted_m_hot_vector[l_index] = 1
                else:
                    # Remove the label as positive and leave it as negative
                    pass
            else:
                # Keep the label as positive
                constrained_predicted_m_hot_vector[l_index] = 1
        else:
            # Keep the label as negative
            pass
    # Return the cleaned multi-hot vector, in which labels which are predicted as positive but their parent is not
    # predicted as positive are removed
    return constrained_predicted_m_hot_vector


def constrained_micro_f1_global_and_per_level(
    targets: list[list[int]],
    predictions: list[list[int]],
    child_parent_indexes: dict[int, int],
    per_level_size: list[int],
) -> dict[str, float]:
    """
    Calculate the constrained micro F1 global and per level.

    :param targets: A list of all the target labels, encoded in a multi-hot format.
    :param predictions: A list of all the predicted labels, encoded in a multi-hot format.
    :param child_parent_indexes: A dictionary in which each key is the index of a child label and the value is the
    index of the parent label.
    :param per_level_size: A list containing the number of labels for each hierarchy level.

    :return: A dictionary containing the constrained micro F1 global and per level.
    """
    # Clean every prediction based on the hierarchy
    predictions = [
        _constrained_output_with_respect_to_hierarchy(
            child_parent_indexes=child_parent_indexes, predicted_m_hot_vector=prediction
        )
        for prediction in predictions
    ]
    # Calculate the micro F1 global and per level, with the cleaned predictions
    micro_f1 = micro_f1_global_and_per_level(targets, predictions, per_level_size)
    return {
        "c-micro_f1_global": micro_f1["micro_f1_global"],
        "c-micro_f1_per_level": micro_f1["micro_f1_per_level"],
    }


def constrained_macro_f1_global_and_per_level(
    targets: list[list[int]],
    predictions: list[list[int]],
    child_parent_indexes: dict[int, int],
    per_level_size: list[int],
) -> dict[str, float]:
    """
    Calculate the constrained macro F1 global and per level.

    :param targets: A list of all the target labels, encoded in a multi-hot format.
    :param predictions: A list of all the predicted labels, encoded in a multi-hot format.
    :param child_parent_indexes: A dictionary in which each key is the index of a child label and the value is the
    index of the parent label.
    :param per_level_size: A list containing the number of labels for each hierarchy level.

    :return: A dictionary containing the constrained macro F1 global and per level.
    """
    predictions = [
        _constrained_output_with_respect_to_hierarchy(child_parent_indexes, prediction)
        for prediction in predictions
    ]
    micro_f1 = macro_f1_global_and_per_level(targets, predictions, per_level_size)
    return {
        "c-macro_f1_global": micro_f1["macro_f1_global"],
        "c-macro_f1_per_level": micro_f1["macro_f1_per_level"],
    }
