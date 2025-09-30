from enum import Enum
import os
import torch
from transformers import PretrainedConfig


class BaseConfig(PretrainedConfig):

    @classmethod
    def from_pretrained(cls, load_directory: str, file_name: str = None, **kwargs):
        file_name = file_name or "config.json"
        config_file_path = os.path.join(load_directory, file_name)
        config = None
        try:
            if os.path.exists(config_file_path):
                config = super().from_pretrained(config_file_path, **kwargs)
                return config
        except OSError:
            print(
                f"Configuration file {config_file_path} not found. "
                f"Loading default configuration."
            )
        return None

    def save_pretrained(self, save_directory: str, file_name: str = None):
        file_name = file_name or "config.json"
        os.makedirs(save_directory, exist_ok=True)
        # Save the configuration to a JSON file
        config_file_path = os.path.join(save_directory, file_name)
        with open(config_file_path, "w") as f:
            f.write(self.to_json_string())

    def get_field(self, name, default=None):
        """
        Returns the value of the field with the given name if it exists,
        otherwise returns the provided default value.
        """
        return getattr(self, name, default)


class DecoderType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    NONE = "none"

    @classmethod
    def from_string(cls, decoder_type: str):
        if decoder_type.lower() == "lstm":
            return cls.LSTM
        elif decoder_type.lower() == "transformer":
            return cls.TRANSFORMER
        elif decoder_type.lower() == "none":
            return cls.NONE
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")


class StructuredGenerativeT5Config(BaseConfig):

    model_type = "structured_generator"
    base_model_prefix = "structured_generator"

    def __init__(
        self,
        decoder_type: str = DecoderType.LSTM.value,
        decoder_tokenizer_path: str = None,
        decoder_input_size=256,
        decoder_hidden_size=512,
        decoder_vocab_size=30522,
        decoder_max_input_length=50,
        decoder_num_layers=1,
        decoder_bos_token_id=0,
        decoder_eos_token_id=0,
        decoder_pad_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type
        self.lstm_tokenizer_path = decoder_tokenizer_path
        self.decoder_input_size = decoder_input_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_max_input_length = decoder_max_input_length
        self.decoder_num_layers = decoder_num_layers
        self.decoder_bos_token_id = decoder_bos_token_id
        self.decoder_eos_token_id = decoder_eos_token_id
        self.decoder_pad_token_id = decoder_pad_token_id
        self.hf_encoder_name = kwargs.get(
            "hf_encoder_name",
            "bert-base-uncased",
        )
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder_dropout = kwargs.get("encoder_dropout", 0.2)

        # Dynamically set any additional fields from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaselineT5Config(BaseConfig):
    model_type = "baseline_t5"
    base_model_prefix = "baseline_t5"

    def __init__(
        self,
        target_type: str = "dfs",
        max_target_length: int = 50,
        max_source_length: int = 512,
        constrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length
        self.constrained = constrained
        self.hf_encoder_name = kwargs.get(
            "hf_encoder_name",
            "bert-base-uncased",
        )

        # Dynamically set any additional fields from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class TrainingConfig(BaseConfig):

    def __init__(
        self,
        batch_size=32,
        adam_lr=5e-5,
        after_warmup_adam_lr=5e-5,
        early_stopping_patience=5,
        max_steps=None,
        max_epochs=1,
        warmup_steps=None,
        warmup_epochs=1,
        accum_iter=1,
        do_linear_lr=False,
        do_scale=False,
        warmup_parameters=None,
        after_warmup_parameters=None,
        length_penalty_weight=None,
        initial_teacher_forcing_ratio=1.0,
        final_teacher_forcing_ratio=0.0,
        label_smoothing=0.0,
        **kwargs,
    ):
        assertion_value = warmup_steps is None or warmup_epochs is not None
        assertion_message = (
            "If warmup_steps is provided, warmup_epochs must also be provided."
        )
        assert assertion_value, assertion_message
        assertion_value = max_epochs is not None or max_steps is not None
        assertion_message = (
            "Either max_epochs or max_steps must be provided, but not both."
        )
        assert assertion_value, assertion_message

        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.adam_lr = adam_lr
        self.after_warmup_adam_lr = after_warmup_adam_lr
        self.early_stopping_patience = early_stopping_patience
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.accum_iter = accum_iter
        self.warmup_parameters = warmup_parameters
        self.after_warmup_parameters = after_warmup_parameters
        self.do_linear_lr = do_linear_lr
        self.do_scale = do_scale
        self.length_penalty_weight = length_penalty_weight
        self.initial_teacher_forcing_ratio = initial_teacher_forcing_ratio
        self.final_teacher_forcing_ratio = final_teacher_forcing_ratio
        self.label_smoothing = label_smoothing

        # Dynamically set any additional fields from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
