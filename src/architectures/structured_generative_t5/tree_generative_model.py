from typing import Optional

from info_nce import InfoNCE
import torch
import torch.nn as nn

from src.architectures.abstract_t5_model import AbstractT5Model
from src.architectures.structured_generative_t5.char_decoder import (
    TransformerCharDecoder,
)
from src.architectures.structured_generative_t5.encoder import Encoder
from src.architectures.structured_generative_t5.lstm_decoder import (
    LstmDecoder,
)
from src.architectures.structured_generative_t5.config import (
    DecoderType,
    StructuredGenerativeT5Config,
    TrainingConfig,
)


class TreeGenerativeModel(AbstractT5Model):
    """
    TreeGenerativeModel dynamically supports LSTM-based or Transformer-based decoding
    for structured generation tasks.

    Args:
        local_folder_path (str): Path to save/load model artifacts.
        config (StructuredGenerationConfig): Configuration object with model hyperparameters.
        label_weights (torch.Tensor): Class weights for the loss function.
        decoder_type (str): Decoder type to use, either 'lstm' or 'transformer'.

    """

    def __init__(
        self,
        local_folder_path: str,
        config: StructuredGenerativeT5Config,
        label_weights: torch.Tensor,
        label_id_descriptions: dict[int, str],
    ):
        super().__init__(
            local_folder_path=local_folder_path,
        )

        self.config: StructuredGenerativeT5Config = config

        # Initialize the encoder
        self.encoder: Encoder = Encoder(
            config=config,
            coupled_decoder_type=DecoderType.from_string(
                config.decoder_type,
            ),
        )

        # Check if encoder requires token_type_ids during forward
        self.encoder_requires_type_ids = (
            "token_type_ids" in self.encoder.encoder_layer.forward.__code__.co_varnames
        )

        if config.decoder_type == DecoderType.LSTM.value:
            self.decoder = LstmDecoder(
                config=config,
            )
        elif config.decoder_type == DecoderType.TRANSFORMER.value:
            self.decoder = TransformerCharDecoder(
                config=config,
                get_label_embeddings=lambda x: self.encoder(**x),
                label_id_descriptions=label_id_descriptions,
            )
        else:
            raise ValueError(
                f"Unsupported decoder_type '{config.decoder_type}'. "
                "Supported types are 'lstm' and 'transformer'."
            )

        self.label_weights = torch.tensor(
            label_weights,
            dtype=torch.float32,
            device=self.get_device(),
        )
        self.to(self.get_device())

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        negative_labels: Optional[torch.LongTensor] = None,
        use_teacher_forcing: bool = True,
        max_generation_length: Optional[int] = None,
        training_config: Optional["TrainingConfig"] = None,
        **kwargs,
    ) -> tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass for training or generation.

        Returns:
            loss (optional): Cross-entropy + length penalty loss if labels are provided.
            probabilities: Softmax probabilities over vocab for each token position.
            predicted_token_ids: Tokens predicted by the decoder.
            negative_labels (optional): A Tensor of shape (batch_size, sequence_length, num_negative_samples) containing
                the identifiers of the negative samples for each position in the sequence.
        """
        # TODO modify the calling method to pass also the tensor containing the negative samples
        assert (
            not use_teacher_forcing or labels is not None
        ), "Teacher forcing enabled but no labels provided."
        assert (
            labels is not None or max_generation_length is not None
        ), "Either labels or max_generation_length must be provided."

        # Move inputs to device
        input_ids = input_ids.to(self.get_device())
        token_type_ids = token_type_ids.to(self.get_device())
        attention_mask = attention_mask.to(self.get_device())

        if labels is not None:
            labels = labels.to(self.get_device())
            max_generation_length = labels.size(1)

        # Encoder forward: get CLS token embedding or pooled output
        cls_token_embedding = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if self.encoder_requires_type_ids else None,
        )

        # Pass the encoder output to the decoder and perform the forward pass to predict the
        # logits of the entire sequence
        logits = self.decoder(
            batch_size=input_ids.size(0),
            encoder_embedding=cls_token_embedding,
            max_generation_length=max_generation_length - 1,
            use_teacher_forcing=False,  # TODO use_teacher_forcing,
            true_labels=labels,
            encoder_attention_mask=attention_mask,
            force_max_generation=True,  # TODO make this a parameter so that it is not True at inference time
        )
        if type(logits) is tuple:
            # If the decoder returns a tuple, extract logits
            logits = logits[1]

        # Calculate predicted tokens by
        predicted_token_ids = torch.argmax(logits, dim=-1)

        loss = None
        if labels is not None:
            labels = labels[:, 1:]
            # Pass the positive labels to the embedding layer of the decoder, to obtain a tensor of shape
            # (batch_size, sequence_length, embedding_dim)
            positive_labels_embeddings = self.decoder.embedding(labels)
            negative_labels = negative_labels[:, 1:, :]
            # Pass the negative labels to the embedding layer of the decoder, to obtain a tensor of shape
            # (batch_size, sequence_length, num_negative_samples, embedding_dim)
            negative_labels_embeddings = self.decoder.embedding(negative_labels)
            # If the second dimension of the logits has one value less than the labels, remove the first element
            # of the labels to match the logits size.
            if logits.size(1) == labels.size(1) + 1:
                logits = logits[:, :-1]

            try:
                # Perform the InfoNCE loss to try to bring the logits of each predicted "token" closer to the
                # embeddings of the positive labels, and farther from the embeddings of the negative labels.
                loss = InfoNCE(
                    negative_mode="paired",
                )
                info_nce_loss = loss(
                    logits.reshape(-1, logits.size(2)),
                    positive_labels_embeddings.reshape(
                        -1, positive_labels_embeddings.size(2)
                    ),
                    negative_labels_embeddings.reshape(
                        -1,
                        negative_labels_embeddings.size(2),
                        negative_labels_embeddings.size(3),
                    ),
                )
                # Perform the usual cross-entropy loss to account for the correct prediction of the next token.
                # However, this loss works in the vocabulary space, while the InfoNCE works in the embedding space.
                ce_loss = nn.CrossEntropyLoss(
                    ignore_index=self.config.decoder_pad_token_id,
                    # weight=self.label_weights,
                    # label_smoothing=(
                    #     training_config.label_smoothing if training_config else 0.0
                    # ),
                )(
                    logits.contiguous().view(-1, logits.size(-1)),
                    labels.contiguous().view(-1),
                )
            except ValueError as e:
                print(labels.shape, logits.shape)
                raise e

            if (
                training_config
                and training_config.length_penalty_weight is not None
                and training_config.length_penalty_weight > 0
            ):
                pad_token_id = self.config.decoder_pad_token_id
                eos_token_id = self.config.decoder_eos_token_id

                def get_effective_lengths(seqs, pad_token_id, eos_token_id):
                    is_eos_or_pad = (seqs == eos_token_id) | (seqs == pad_token_id)
                    first_eos_or_pad = is_eos_or_pad.float().cumsum(dim=1).eq(1)
                    valid_mask = (~is_eos_or_pad) & (
                        ~first_eos_or_pad.cumsum(dim=1).bool()
                    )
                    return valid_mask.sum(dim=1).float()

                true_lengths = get_effective_lengths(
                    labels,
                    pad_token_id,
                    eos_token_id,
                )
                pred_lengths = get_effective_lengths(
                    predicted_token_ids,
                    pad_token_id,
                    eos_token_id,
                )

                l1_length_penalty = torch.abs(true_lengths - pred_lengths).mean()
                loss = (
                    ce_loss
                    + info_nce_loss
                    + training_config.length_penalty_weight * l1_length_penalty
                )
            else:
                loss = ce_loss + info_nce_loss

        return loss, logits, predicted_token_ids
