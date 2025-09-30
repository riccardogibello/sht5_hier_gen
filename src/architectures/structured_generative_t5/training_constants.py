from enum import Enum
from typing import Any
from src.utils.constants import DEBUG

EXPERIMENT_NAMES = [
    "blurb_genre_collection",
    "web_of_science",
    "imdrf_mapping",
    "emdn_gmdn",
]


class ProgramParameter(Enum):
    # GENERIC PARAMETERS
    EXPERIMENT_NAMES = "experiment_names"
    BASE_DATA_FOLDER = "base_data_folder"
    MODEL_NAME = "model_name"
    DECODER_TYPE = "decoder_type"
    HF_MODEL_NAME = "hf_model_name"
    PER_LEVEL_CHARS = "per_level_chars"
    CUMULATIVE = "cumulative"
    # TRAINING PARAMETERS
    WARMUP_EPOCHS = "warmup_epochs"
    MAX_EPOCHS = "max_epochs"
    BATCH_SIZE = "batch_size"
    ADAM_LR = "adam_lr"
    DO_LINEAR_INCREASING_LR = "do_linear_increasing_lr"
    AFTER_WARMUP_ADAM_LR = "after_warmup_adam_lr"
    EARLY_STOPPING_PATIENCE = "early_stopping_patience"
    ACCUMULATED_ITERATIONS = "accumulated_iterations"
    DO_SCALED_TRAINING = "do_scaled_training"
    WARMUP_PARAMETERS = "warmup_parameters"
    AFTER_WARMUP_PARAMETERS = "after_warmup_parameters"

    @staticmethod
    def get_parameter(
        program_parameter: "ProgramParameter",
        **kwargs: Any,
    ) -> Any:
        if program_parameter == ProgramParameter.EXPERIMENT_NAMES:
            return [
                "blurb_genre_collection",
            ]
        elif program_parameter == ProgramParameter.BASE_DATA_FOLDER:
            return "."
        elif program_parameter == ProgramParameter.MODEL_NAME:
            return "t5_baseline"
        elif program_parameter == ProgramParameter.DECODER_TYPE:
            return "transformer"
        elif program_parameter == ProgramParameter.HF_MODEL_NAME:
            return "google-t5/t5-small"
        elif program_parameter == ProgramParameter.PER_LEVEL_CHARS:
            return 1
        elif program_parameter == ProgramParameter.CUMULATIVE:
            return False
        elif program_parameter == ProgramParameter.ACCUMULATED_ITERATIONS:
            return 1
        elif program_parameter == ProgramParameter.EARLY_STOPPING_PATIENCE:
            return 5
        elif program_parameter == ProgramParameter.DO_SCALED_TRAINING:
            return False
        elif program_parameter == ProgramParameter.DO_LINEAR_INCREASING_LR:
            return False
        elif program_parameter == ProgramParameter.WARMUP_EPOCHS:
            return 2
        elif program_parameter == ProgramParameter.BATCH_SIZE:
            return 16
        elif program_parameter == ProgramParameter.ADAM_LR:
            return 1e-05
        elif program_parameter == ProgramParameter.AFTER_WARMUP_ADAM_LR:
            return 1e-05
        else:
            # TRAINING PARAMETERS
            model_name = kwargs.get("model_name", None)
            decoder_type = kwargs.get("decoder_type", None)
            if model_name is None or decoder_type is None:
                raise ValueError(
                    "Model name and decoder type must be provided for training parameters."
                )
            else:
                if program_parameter == ProgramParameter.MAX_EPOCHS:
                    if model_name == "t5_baseline":
                        return 50 if not DEBUG else 1
                    else:
                        return 50 if not DEBUG else 1
                elif program_parameter == ProgramParameter.WARMUP_PARAMETERS:
                    if model_name == "t5_baseline":
                        return []
                    else:
                        return ["decoder"]
                elif program_parameter == ProgramParameter.AFTER_WARMUP_PARAMETERS:
                    return []  # Train the full model after warmup
