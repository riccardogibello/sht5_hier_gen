from enum import Enum
from typing import Optional
import torch.nn as nn


class ModelName(Enum):
    T5_BASELINE = "t5_baseline"
    STRUCTURED_GENERATIVE_T5 = "sgt5"

    @staticmethod
    def get_model(model_name: str) -> "ModelName":
        for model in ModelName:
            if model.value == model_name:
                return model
        raise ValueError(f"Model {model_name} not found.")


def set_mode(
    model: nn.Module,
    parameter_names: Optional[list[str]] = None,
    do_train: bool = True,
):
    model.train(do_train)

    parameter_names = parameter_names or []

    # Freeze any parameters in the model except the classifier
    for name, param in model.named_parameters():
        is_parameter_to_train = (
            any([name.__contains__(p_name) for p_name in parameter_names])
            or len(parameter_names) == 0
        )
        if do_train and is_parameter_to_train:
            requires_grad = True
            print(f"Parameter '{name}' requires_grad set to {param.requires_grad}")
        else:
            requires_grad = False
        param.requires_grad = requires_grad


def print_model_parameters(model):
    def format_large_number(value):
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        else:
            return str(value)

    print("====================================")
    total_params = 0
    for name, param in model.named_parameters():
        current_layer_params = param.numel()
        total_params += current_layer_params
        if param.requires_grad:
            print(f"{name}: {format_large_number(current_layer_params)} (trainable)")
            pass
        else:
            print(f"{name}: {format_large_number(current_layer_params)} (frozen)")
            pass
    print(f"Total parameters: {format_large_number(total_params)}")
    print("====================================")
