import argparse
import os

from src.architectures.baseline_t5.training import (
    train_model as train_baseline_t5_model,
)
from src.architectures.structured_generative_t5.config import DecoderType

from src.architectures.structured_generative_t5.training import (
    train_model as train_sgt5_model,
)
from src.evaluation.metrics import compute_metrics
from src.utils.constants import (
    LABEL_FILE_NAME,
    MODELS_FOLDER_NAME,
    ORIGINAL_DATA_FOLDER_NAME,
)
from src.utils.model_utils import ModelName


def run_experiments(
    experiment_names: list[str],
    base_data_folder_path: str,
    model_name: ModelName,
    hf_model_name: str,
    per_level_chars: int,
    arguments: argparse.ArgumentParser,
    cumulative: bool = False,
    decoder_type: DecoderType = DecoderType.NONE,
    use_latents: bool = False,
):
    # For each experiment name
    for experiment_name in experiment_names:
        # TODO this should account for ShT5 used with latent representations
        if (
            cumulative and model_name != ModelName.STRUCTURED_GENERATIVE_T5
        ) or per_level_chars > 1:
            if experiment_name == "imdrf_mapping":
                first_level_chars = 3
            else:
                first_level_chars = 1
        else:
            first_level_chars = per_level_chars
        # Get the paths to the experiment's base data folder and the output file for the metrics
        experiment_base_data_folder = os.path.join(
            base_data_folder_path,
            experiment_name,
        )
        # Run the right pipeline for the current model
        if model_name.value == ModelName.T5_BASELINE.value:
            train_baseline_t5_model(
                experiment_data_folder=experiment_base_data_folder,
                model_name=model_name,
                decoder_type=decoder_type,
                hf_model_name=hf_model_name,
                first_level_chars=first_level_chars,
                per_level_chars=per_level_chars,
                args=arguments,
            )
        elif model_name.value == ModelName.STRUCTURED_GENERATIVE_T5.value:
            train_sgt5_model(
                experiment_data_folder=experiment_base_data_folder,
                model_name=model_name,
                hf_model_name=hf_model_name,
                decoder_type=decoder_type,
                first_level_chars=first_level_chars,
                per_level_chars=per_level_chars,
                args=arguments,
                use_latents=use_latents,
            )
        else:
            print(f"Model {model_name.value} is not supported for training. ")

        compute_metrics(
            base_data_folder_path=base_data_folder_path,
            models_folder_path=os.path.join(
                experiment_base_data_folder,
                MODELS_FOLDER_NAME,
            ),
            labels_file_path=os.path.join(
                experiment_base_data_folder,
                ORIGINAL_DATA_FOLDER_NAME,
                LABEL_FILE_NAME,
            ),
            first_level_chars=3 if experiment_name == "imdrf_mapping" else 1,
            per_level_chars=2,
            model_name=model_name,
        )
