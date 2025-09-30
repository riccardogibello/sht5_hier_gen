import argparse

from src.architectures.structured_generative_t5.config import DecoderType
from src.architectures.structured_generative_t5.training_constants import (
    EXPERIMENT_NAMES,
    ProgramParameter,
)
from src.utils.model_utils import ModelName


def _add_generic_args(
    parser: argparse.ArgumentParser,
):
    parser.add_argument(
        f"--{ProgramParameter.BASE_DATA_FOLDER.value}",
        type=str,
        required=True,
        help="Base data folder for the experiments",
    )
    parser.add_argument(
        f"--{ProgramParameter.EXPERIMENT_NAMES.value}",
        type=str,
        nargs="+",
        required=True,
        choices=EXPERIMENT_NAMES,
        help="List of experiment names",
    )
    parser.add_argument(
        f"--{ProgramParameter.MODEL_NAME.value}",
        type=str,
        required=True,
        choices=[model.value for model in ModelName],
        help="Model name to be used for the experiments",
    )
    parser.add_argument(
        f"--{ProgramParameter.DECODER_TYPE.value}",
        type=str,
        required=False,
        default=ProgramParameter.get_parameter(
            ProgramParameter.DECODER_TYPE,
        ),
        choices=[
            DecoderType.LSTM.value,
            DecoderType.TRANSFORMER.value,
            DecoderType.NONE.value,
        ],
        help="Type of decoder to use (lstm, transformer, none)",
    )
    parser.add_argument(
        f"--{ProgramParameter.PER_LEVEL_CHARS.value}",
        type=int,
        required=False,
        default=ProgramParameter.get_parameter(
            ProgramParameter.PER_LEVEL_CHARS,
        ),
        help="The number of characters per level to use for the decoder",
    )
    parser.add_argument(
        f"--{ProgramParameter.CUMULATIVE.value}",
        action="store_true",
        default=ProgramParameter.get_parameter(
            ProgramParameter.CUMULATIVE,
        ),
        help="Whether to use cumulative labels for the decoder",
    )
    parser.add_argument(
        f"--{ProgramParameter.HF_MODEL_NAME.value}",
        type=str,
        default=ProgramParameter.get_parameter(
            ProgramParameter.HF_MODEL_NAME,
        ),
        help="Hugging Face model name",
    )


def _add_training_args(
    parser: argparse.ArgumentParser,
):
    model_name = ProgramParameter.get_parameter(
        ProgramParameter.MODEL_NAME,
    )
    decoder_type = ProgramParameter.get_parameter(
        ProgramParameter.DECODER_TYPE,
    )
    parser.add_argument(
        f"--{ProgramParameter.WARMUP_EPOCHS.value}",
        type=int,
        default=ProgramParameter.get_parameter(
            ProgramParameter.WARMUP_EPOCHS,
        ),
        help="Number of warmup epochs for the training",
    )
    parser.add_argument(
        f"--{ProgramParameter.MAX_EPOCHS.value}",
        type=int,
        default=ProgramParameter.get_parameter(
            ProgramParameter.MAX_EPOCHS,
            model_name=model_name,
            decoder_type=decoder_type,
        ),
        help="Maximum number of epochs for the training",
    )
    parser.add_argument(
        f"--{ProgramParameter.BATCH_SIZE.value}",
        type=int,
        default=ProgramParameter.get_parameter(
            ProgramParameter.BATCH_SIZE,
        ),
        help="Batch size for the training",
    )
    parser.add_argument(
        f"--{ProgramParameter.ADAM_LR.value}",
        type=float,
        default=ProgramParameter.get_parameter(
            ProgramParameter.ADAM_LR,
        ),
        help="Learning rate for the Adam optimizer",
    )
    parser.add_argument(
        f"--{ProgramParameter.DO_LINEAR_INCREASING_LR.value}",
        action="store_true",
        default=ProgramParameter.get_parameter(
            ProgramParameter.DO_LINEAR_INCREASING_LR,
        ),
        help="Whether to use linear increasing learning rate",
    )
    parser.add_argument(
        f"--{ProgramParameter.AFTER_WARMUP_ADAM_LR.value}",
        type=float,
        default=ProgramParameter.get_parameter(
            ProgramParameter.AFTER_WARMUP_ADAM_LR,
        ),
        help="Learning rate after warmup for the Adam optimizer",
    )
    parser.add_argument(
        f"--{ProgramParameter.EARLY_STOPPING_PATIENCE.value}",
        type=int,
        default=ProgramParameter.get_parameter(
            ProgramParameter.EARLY_STOPPING_PATIENCE,
        ),
        help="Early stopping patience for the training",
    )
    parser.add_argument(
        f"--{ProgramParameter.ACCUMULATED_ITERATIONS.value}",
        type=int,
        default=ProgramParameter.get_parameter(
            ProgramParameter.ACCUMULATED_ITERATIONS,
        ),
        help="Number of accumulated iterations for the training",
    )
    parser.add_argument(
        f"--{ProgramParameter.DO_SCALED_TRAINING.value}",
        action="store_true",
        default=ProgramParameter.get_parameter(
            ProgramParameter.DO_SCALED_TRAINING,
        ),
        help="Whether to use scaled training",
    )
    parser.add_argument(
        f"--{ProgramParameter.WARMUP_PARAMETERS.value}",
        type=str,
        nargs="+",
        default=ProgramParameter.get_parameter(
            ProgramParameter.WARMUP_PARAMETERS,
            model_name=model_name,
            decoder_type=decoder_type,
        ),
        help="List of parameters to warmup during training",
    )
    parser.add_argument(
        f"--{ProgramParameter.AFTER_WARMUP_PARAMETERS.value}",
        type=str,
        nargs="+",
        default=ProgramParameter.get_parameter(
            ProgramParameter.AFTER_WARMUP_PARAMETERS,
            model_name=model_name,
            decoder_type=decoder_type,
        ),
        help="List of parameters to use after warmup during training",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments with specified models."
    )

    _add_generic_args(parser)
    _add_training_args(parser)

    return parser.parse_args()
