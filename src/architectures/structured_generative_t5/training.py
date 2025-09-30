import argparse
import os
import re

from transformers import AutoTokenizer

from src.architectures.structured_generative_t5.training_constants import (
    ProgramParameter,
)
from src.architectures.structured_generative_t5.tree_generative_model import (
    TreeGenerativeModel,
)
from src.data_preprocessing.data_loading import load_data
from src.architectures.structured_generative_t5.config import (
    DecoderType,
    StructuredGenerativeT5Config,
    TrainingConfig,
)
from src.data_preprocessing.tokenizer import HierarchicalLabelTokenizerFast
from src.architectures.training import test, train
from src.evaluation.output import store_mappings
from src.utils.constants import (
    MAX_LEVEL,
    CONFIG_FILE_NAME,
    LABEL_FILE_NAME,
    MODELS_FOLDER_NAME,
    ORIGINAL_DATA_FOLDER_NAME,
    TOKENIZER_FOLDER_NAME,
    TRAIN_CONFIG_FILE_NAME,
)
from src.utils.model_utils import ModelName, print_model_parameters


def train_model(
    experiment_data_folder: str,
    model_name: ModelName,
    decoder_type: DecoderType,
    hf_model_name: str,
    first_level_chars: int,
    per_level_chars: int,
    args: argparse.ArgumentParser,
):
    labels_file_path = os.path.join(
        experiment_data_folder,
        ORIGINAL_DATA_FOLDER_NAME,
        LABEL_FILE_NAME,
    )
    cumulative = getattr(
        args,
        ProgramParameter.CUMULATIVE.value,
        ProgramParameter.get_parameter(
            ProgramParameter.CUMULATIVE,
        ),
    )
    hf_model_name_cleaned = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        hf_model_name.split("/")[-1],
    )
    model_folder_path = os.path.join(
        experiment_data_folder,
        MODELS_FOLDER_NAME,
        model_name.value
        + f"_{hf_model_name_cleaned}"
        + f"_{decoder_type.value}_decoder_{per_level_chars}pl_cum_{cumulative}",
    )
    cached_dataset_file_name = f"{model_name.value}_{hf_model_name_cleaned}_{per_level_chars}pl_cum_{cumulative}_cached_dataset.pt"
    tokenizer_folder_path = os.path.join(
        model_folder_path,
        TOKENIZER_FOLDER_NAME,
    )

    max_level = MAX_LEVEL
    sg_tokenizer = HierarchicalLabelTokenizerFast(
        labels_csv_file_path=labels_file_path,
        max_level=max_level,
        tokenizer_dir=tokenizer_folder_path,
        per_level_chars=per_level_chars,
    )
    sg_tokenizer.save_pretrained(
        tokenizer_folder_path,
    )

    original_tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
    )
    if original_tokenizer.model_max_length > 512:
        original_tokenizer.model_max_length = 512

    batch_size = getattr(
        args,
        ProgramParameter.BATCH_SIZE.value,
        ProgramParameter.get_parameter(
            ProgramParameter.BATCH_SIZE,
        ),
    )

    train_dataloader, validation_dataloader, test_dataloader, label_weights = load_data(
        input_tokenizer=original_tokenizer,
        output_tokenizer=sg_tokenizer,
        max_output_length=50,
        experiment_data_folder=experiment_data_folder,
        output_file_name=cached_dataset_file_name,
        first_level_chars=first_level_chars,
        per_level_chars=per_level_chars,
        max_level=max_level,
        batch_size=batch_size,
        cumulative=cumulative,
        model_name=model_name,
    )

    config = StructuredGenerativeT5Config.from_pretrained(
        load_directory=model_folder_path,
        file_name=CONFIG_FILE_NAME,
    )
    train_config = TrainingConfig.from_pretrained(
        load_directory=model_folder_path,
        file_name=TRAIN_CONFIG_FILE_NAME,
    )
    if config is None:
        os.makedirs(model_folder_path, exist_ok=True)
        config = StructuredGenerativeT5Config(
            decoder_type=decoder_type.value,
            decoder_tokenizer_path=os.path.join(
                experiment_data_folder,
                TOKENIZER_FOLDER_NAME,
            ),
            decoder_vocab_size=len(sg_tokenizer.get_vocab()),
            decoder_hidden_size=512,
            decoder_input_size=512,
            decoder_num_layers=1,
            hf_encoder_name=hf_model_name,
            decoder_bos_token_id=sg_tokenizer.bos_token_id,
            decoder_eos_token_id=sg_tokenizer.eos_token_id,
            decoder_pad_token_id=sg_tokenizer.pad_token_id,
        )
        config.save_pretrained(
            save_directory=model_folder_path,
            file_name=CONFIG_FILE_NAME,
        )
    if train_config is None:
        os.makedirs(
            model_folder_path,
            exist_ok=True,
        )

        max_epochs = getattr(
            args,
            ProgramParameter.MAX_EPOCHS.value,
            ProgramParameter.get_parameter(
                ProgramParameter.MAX_EPOCHS,
                model_name=model_name.value,
                decoder_type=decoder_type.value,
            ),
        )
        warmup_epochs = getattr(
            args,
            ProgramParameter.WARMUP_EPOCHS.value,
            ProgramParameter.get_parameter(
                ProgramParameter.WARMUP_EPOCHS,
            ),
        )
        accumulated_iterations = getattr(
            args,
            ProgramParameter.ACCUMULATED_ITERATIONS.value,
            ProgramParameter.get_parameter(
                ProgramParameter.ACCUMULATED_ITERATIONS,
            ),
        )
        do_scaled_training = getattr(
            args,
            ProgramParameter.DO_SCALED_TRAINING.value,
            ProgramParameter.get_parameter(
                ProgramParameter.DO_SCALED_TRAINING,
            ),
        )
        do_linear_increasing_lr = getattr(
            args,
            ProgramParameter.DO_LINEAR_INCREASING_LR.value,
            ProgramParameter.get_parameter(
                ProgramParameter.DO_LINEAR_INCREASING_LR,
            ),
        )
        warmup_parameters = getattr(
            args,
            ProgramParameter.WARMUP_PARAMETERS.value,
            ProgramParameter.get_parameter(
                ProgramParameter.WARMUP_PARAMETERS,
                model_name=model_name.value,
                decoder_type=decoder_type.value,
            ),
        )
        after_warmup_parameters = getattr(
            args,
            ProgramParameter.AFTER_WARMUP_PARAMETERS.value,
            ProgramParameter.get_parameter(
                ProgramParameter.AFTER_WARMUP_PARAMETERS,
                model_name=model_name.value,
                decoder_type=decoder_type.value,
            ),
        )
        train_config = TrainingConfig(
            batch_size=batch_size,
            adam_lr=getattr(
                args,
                ProgramParameter.ADAM_LR.value,
                ProgramParameter.get_parameter(
                    ProgramParameter.ADAM_LR,
                ),
            ),
            after_warmup_adam_lr=getattr(
                args,
                ProgramParameter.AFTER_WARMUP_ADAM_LR.value,
                ProgramParameter.get_parameter(
                    ProgramParameter.AFTER_WARMUP_ADAM_LR,
                ),
            ),
            early_stopping_patience=getattr(
                args,
                ProgramParameter.EARLY_STOPPING_PATIENCE.value,
                ProgramParameter.get_parameter(
                    ProgramParameter.EARLY_STOPPING_PATIENCE
                ),
            ),
            max_steps=len(train_dataloader) * max_epochs,
            max_epochs=max_epochs,
            warmup_steps=len(train_dataloader) * warmup_epochs,
            warmup_epochs=warmup_epochs,
            accum_iter=accumulated_iterations,
            do_linear_lr=do_linear_increasing_lr,
            do_scale=do_scaled_training,
            warmup_parameters=warmup_parameters,
            after_warmup_parameters=after_warmup_parameters,
            length_penalty_weight=0.0,
            initial_teacher_forcing_ratio=1.0,
            final_teacher_forcing_ratio=1.0,
            label_smoothing=0.0,
        )
        train_config.save_pretrained(
            save_directory=model_folder_path,
            file_name=TRAIN_CONFIG_FILE_NAME,
        )

    model = TreeGenerativeModel(
        config=config,
        local_folder_path=model_folder_path,
        label_weights=label_weights,
    )
    train(
        model=model,
        training_config=train_config,
        train_dataloader=train_dataloader,
        model_folder_path=model_folder_path,
        val_dataloader=validation_dataloader,
        store_mappings_callback=lambda file_folder, file_name, src_texts, true_label_tensor, predicted_logits: store_mappings(
            file_folder=file_folder,
            file_name=file_name,
            src_texts=src_texts,
            true_label_tensor=true_label_tensor,
            predicted_logits_or_indices=predicted_logits,
            input_tokenizer=original_tokenizer,
            output_tokenizer=sg_tokenizer,
        ),
    )
    test(
        model=model,
        test_dataloader=test_dataloader,
        training_config=train_config,
        model_folder_path=model_folder_path,
        store_mappings_callback=lambda file_folder, file_name, src_texts, true_label_tensor, predicted_logits: store_mappings(
            file_folder=file_folder,
            file_name=file_name,
            src_texts=src_texts,
            true_label_tensor=true_label_tensor,
            predicted_logits_or_indices=predicted_logits,
            input_tokenizer=original_tokenizer,
            output_tokenizer=sg_tokenizer,
        ),
        use_cache=True,
    )
