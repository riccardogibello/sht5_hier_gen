import math


def compute_lr_factor(
        step: int,
        warmup_steps: int,
        max_steps: int,
) -> float:
    """
    Calculate the learning rate factor for the given step using a linear increase during warmup and a logarithmic decrease after warmup.

    :param step: The current step.
    :param warmup_steps: The number of warmup steps in which the learning rate increases linearly.
    :param max_steps: The maximum number of epochs.

    :return: The learning rate factor for the given step.
    """
    # Ensure step is at least 1 to avoid division by zero
    step = max(step, 1)

    # Calculate the final learning rate
    if step < warmup_steps:
        learning_rate = step / warmup_steps
    else:
        # Logarithmic decrease after warmup
        learning_rate = 1 - math.pow((step - warmup_steps) / (max_steps - warmup_steps), 2)

    return learning_rate
