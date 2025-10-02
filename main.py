import os
import re

from src.architectures.structured_generative_t5.config import DecoderType
from src.architectures.structured_generative_t5.training_constants import (
    ProgramParameter,
)
from src.emissions_tracker import tracked_function
from src.run import run_experiments
from src.utils.model_utils import ModelName
from src.utils.parsing_args import parse_args

# EXAMPLE USAGE:
# python main.py --experiment_names imdrf_mapping --base_data_folder ./data --model_name t5_baseline --decoder_type transformer --hf_model_name google-t5/t5-small --per_level_chars 1 --cumulative


def get_arg(args, param: ProgramParameter) -> any:
    """Helper to get argument with fallback to default."""
    return getattr(
        args,
        param.value,
        ProgramParameter.get_parameter(
            param,
        ),
    )


def normalize_path(path: str) -> str:
    """Normalize path separators for the current OS."""
    return path.replace("/", os.sep).replace("\\", os.sep)


if __name__ == "__main__":
    args = parse_args()

    experiment_names: list[str] = get_arg(args, ProgramParameter.EXPERIMENT_NAMES)
    base_data_folder: str = normalize_path(
        get_arg(args, ProgramParameter.BASE_DATA_FOLDER)
    )
    model_name: ModelName = ModelName.get_model(
        get_arg(args, ProgramParameter.MODEL_NAME)
    )
    use_latents: bool = get_arg(args, ProgramParameter.USE_LATENTS)
    decoder_type = DecoderType.from_string(get_arg(args, ProgramParameter.DECODER_TYPE))
    hf_model_name: str = get_arg(args, ProgramParameter.HF_MODEL_NAME)
    per_level_chars: int = int(get_arg(args, ProgramParameter.PER_LEVEL_CHARS))
    cumulative = get_arg(args, ProgramParameter.CUMULATIVE)

    emissions_path = os.path.join(base_data_folder, "_emissions")

    hf_model_name_cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", hf_model_name.split("/")[-1])
    suffix = (
        f"_{decoder_type.value}_decoder_{per_level_chars}pl_cum_{cumulative}_training"
    )

    for experiment_name in experiment_names:
        tmp_experiment_names = [experiment_name]
        project_name = (
            f"{experiment_name}__{model_name.value}_{hf_model_name_cleaned}{suffix}"
        )
        run_experiments(
            tmp_experiment_names,
            base_data_folder,
            model_name,
            hf_model_name,
            per_level_chars,
            args,
            cumulative,
            decoder_type,
            use_latents,
        )
        """tracked_function(
            _command=run_experiments,
            _output_dir=str(emissions_path),
            _interval=10,
            _project_name=project_name,
            _args=(
                tmp_experiment_names,
                base_data_folder,
                model_name,
                hf_model_name,
                per_level_chars,
                args,
                cumulative,
                decoder_type,
            ),
        )"""
