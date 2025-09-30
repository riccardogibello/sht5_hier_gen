from torch import nn
import torch
from transformers import AutoModel
from src.architectures.structured_generative_t5.config import (
    DecoderType,
    StructuredGenerativeT5Config,
)
from src.utils.constants import RETURN_CLS_TOKEN_FIELD


class Encoder(nn.Module):
    def __init__(
        self,
        config: StructuredGenerativeT5Config,
        coupled_decoder_type: DecoderType,
    ):
        super().__init__()
        self.config = config
        self.coupled_decoder_type = coupled_decoder_type

        if "bert" in config.hf_encoder_name.lower():
            self.encoder_layer = AutoModel.from_pretrained(
                config.hf_encoder_name,
            )
        elif "t5" in config.hf_encoder_name.lower():
            self.encoder_layer = AutoModel.from_pretrained(
                config.hf_encoder_name,
            ).encoder
        else:
            raise ValueError(f"Unsupported model name: {config.hf_encoder_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        if self.encoder_layer.config.hidden_size != config.decoder_hidden_size:
            self.lower_proj = nn.Linear(
                self.encoder_layer.config.hidden_size, config.decoder_hidden_size
            )
            nn.init.xavier_uniform_(self.lower_proj.weight)
            self.dropout = nn.Dropout(config.encoder_dropout)
            self.relu = nn.ReLU()
        else:
            self.lower_proj = self.dropout = self.relu = nn.Identity()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder_layer(**encoder_kwargs)
        # Decide what to return based on config
        if self.config.get_field(
            RETURN_CLS_TOKEN_FIELD,
            True if self.coupled_decoder_type == DecoderType.LSTM else False,
        ):
            x = outputs.last_hidden_state[:, 0, :]
            x = self.lower_proj(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x
        else:
            x = outputs.last_hidden_state
            x = self.lower_proj(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x
