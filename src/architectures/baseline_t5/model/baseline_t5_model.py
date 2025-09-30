from typing import Optional

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from transformers import T5Config
from transformers.generation.utils import GenerateOutput


from src.architectures.abstract_t5_model import AbstractT5Model
from src.architectures.structured_generative_t5.config import (
    BaselineT5Config,
    TrainingConfig,
)


class BaselineT5(AbstractT5Model, T5ForConditionalGeneration):

    def __init__(
        self,
        local_folder_path: str,
        hf_model_name,
        config: BaselineT5Config,
    ):
        # Load the configuration of the model from the Hugging Face library
        pretrained_config = T5Config.from_pretrained(hf_model_name)
        # Initialize the model with the configuration
        T5ForConditionalGeneration.__init__(
            self,
            pretrained_config,
        )
        AbstractT5Model.__init__(
            self,
            local_folder_path=local_folder_path,
        )

        # Set the prefix to be used to prepend the input text
        self.task_prefix = "Classify:"
        # Set the maximum length of the target text
        self.max_target_length = config.max_target_length
        # Get the tokenizer and the model from the pretrained models of the Hugging Face library
        self.tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(hf_model_name)
        # Set the model configuration
        self.config = self.model.config
        # Initialize the dynamic vocabulary cache, which is a dictionary to be passed to the constrained generation
        # mixin to store the dynamic vocabulary of the model. This helps to adhere to the constraints of the hierarchy
        self.dynamic_vocab_cache = {}
        self.constrained = config.constrained

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        use_teacher_forcing: bool = True,
        max_generation_length: Optional[int] = None,
        training_config: Optional["TrainingConfig"] = None,
        **kwargs,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)
            max_generation_length = labels.size(1)
        elif max_generation_length is None:
            max_generation_length = self.max_target_length

        if use_teacher_forcing:
            if labels is None:
                raise ValueError("Labels must be provided when using teacher forcing.")

            # Clone labels and replace padding tokens with -100 to ignore them in loss
            labels = labels.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            results = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = results.loss
            logits = results.logits
            predicted_token_ids = torch.argmax(logits, dim=-1)
        else:
            # Generate predictions without teacher forcing (e.g., at inference)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_generation_length,
                use_cache=True,
                **kwargs,
            )

            predicted_token_ids = generated_ids
            loss = torch.tensor(
                0.0,
                device=self.device,
            )
            logits = torch.zeros(
                generated_ids.size(0),
                generated_ids.size(1),
                device=self.device,
            ).shape

        return loss, logits, predicted_token_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> tuple[
        torch.LongTensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[str]],
        list[list[str]],
    ]:
        """
        Generation using Hugging Face's generate API with attention outputs.
        Returns:
            - generated_ids: [batch_size, max_length]
            - decoder_attentions: List[num_layers][batch_size, num_heads, tgt_len, tgt_len]
            - cross_attentions: List[num_layers][batch_size, num_heads, tgt_len, src_len]
            - input_tokens: List[List[str]] - decoded input tokens
            - output_tokens: List[List[str]] - decoded output tokens
        """
        max_length = max_length or self.max_target_length

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs,
        )

        generated_ids = output.sequences
        decoder_attentions = (
            output.decoder_attentions
        )  # List[num_layers][batch, heads, tgt_len, tgt_len]
        cross_attentions = (
            output.cross_attentions
        )  # List[num_layers][batch, heads, tgt_len, src_len]

        # Decode input and output tokens (as list of token strings)
        input_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        output_tokens = [
            self.tokenizer.convert_ids_to_tokens(ids) for ids in generated_ids
        ]

        return (
            generated_ids,
            decoder_attentions,
            cross_attentions,
            input_tokens,
            output_tokens,
        )
