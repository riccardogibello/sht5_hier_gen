from copy import deepcopy
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm
from transformers import T5ForConditionalGeneration

from src.architectures.structured_generative_t5.config import (
    StructuredGenerativeT5Config,
)


class TransformerCharDecoder(nn.Module):

    def __init__(
        self,
        config: StructuredGenerativeT5Config,
        get_label_embeddings: Callable[[torch.Tensor], torch.Tensor] = None,
        label_id_descriptions: dict[int, str] = None,
    ):
        super().__init__()

        self.config = config
        self.hidden_size = config.decoder_hidden_size
        self.vocab_size = config.decoder_vocab_size
        self.pad_token_id = config.decoder_pad_token_id
        self.bos_token_id = config.decoder_bos_token_id
        self.eos_token_id = config.decoder_eos_token_id
        self.max_length = config.decoder_max_input_length
        self.get_label_embeddings = get_label_embeddings
        self.label_id_descriptions = label_id_descriptions

        # Load original pretrained T5 encoder for config
        original_model = T5ForConditionalGeneration.from_pretrained(
            config.hf_encoder_name
        )
        new_config = deepcopy(original_model.config)
        new_config.decoder_start_token_id = self.bos_token_id
        new_config.vocab_size = self.vocab_size
        new_config.num_decoder_layers = config.decoder_num_layers
        new_config.pad_token_id = self.pad_token_id
        new_config.eos_token_id = self.eos_token_id
        new_config.is_decoder = True
        new_config.use_cache = False

        self.dropout = nn.Dropout(new_config.dropout_rate)

        # === Embeddings ===
        self.output_projection = None
        self.final_layer_norm = None
        if self.get_label_embeddings is None:
            # Standard embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.hidden_size,
                padding_idx=self.pad_token_id,
            )
            self.output_projection = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False
            )
            self.tie_weights()
            self.final_layer_norm = T5LayerNorm(
                self.hidden_size, eps=new_config.layer_norm_epsilon
            )
        else:
            self.init_label_embeddings()

        # === Decoder blocks ===
        self.decoder_blocks = nn.ModuleList(
            [
                T5Block(new_config, layer_idx=i, has_relative_attention_bias=(i == 0))
                for i in range(new_config.num_decoder_layers)
            ]
        )

        # Initialize weights
        self._init_weights()

        # Cleanup
        del original_model
        torch.cuda.empty_cache()

    # -------------------------
    # Helper methods
    # -------------------------
    def get_device(self):
        return next(self.parameters()).device

    def tie_weights(self):
        if self.output_projection is not None:
            self.output_projection.weight = self.embedding.weight

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.final_layer_norm is not None:
            nn.init.ones_(self.final_layer_norm.weight)
        for block in self.decoder_blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if hasattr(module, "bias") and module.bias is not None:
                        nn.init.zeros_(module.bias)

    def init_label_embeddings(self):
        # TODO call this every X steps to update the code embeddings used as input
        if self.get_label_embeddings is None:
            raise ValueError(
                "Cannot initialize label embeddings without an encoder callback."
            )

        # Extract the list of descriptions to be encoded
        if self.label_id_descriptions is None:
            raise ValueError("Label id descriptions are not provided.")

        total_descriptions = len(self.label_id_descriptions["input_ids"])
        batch_size = 32
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, total_descriptions, batch_size):
                current_input_ids = self.label_id_descriptions["input_ids"][
                    i : i + batch_size
                ]
                current_attention_mask = self.label_id_descriptions["attention_mask"][
                    i : i + batch_size
                ]
                if "token_type_ids" in self.label_id_descriptions:
                    current_token_type_ids = self.label_id_descriptions[
                        "token_type_ids"
                    ][i : i + batch_size]
                    batch_descriptions = {
                        "input_ids": torch.tensor(
                            current_input_ids,
                        ),
                        "attention_mask": torch.tensor(
                            current_attention_mask,
                        ),
                        "token_type_ids": torch.tensor(
                            current_token_type_ids,
                        ),
                    }
                else:
                    batch_descriptions = {
                        "input_ids": torch.tensor(
                            current_input_ids,
                        ),
                        "attention_mask": torch.tensor(
                            current_attention_mask,
                        ),
                    }
                # Use the mean of the last hidden state as embedding
                embeddings = self.get_label_embeddings(batch_descriptions).mean(dim=1)
                all_embeddings.append(embeddings.cpu())
        # Concatenate all embeddings in a single tensor
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.embedding = nn.Embedding.from_pretrained(
            all_embeddings,
            freeze=True,
        )
        self.build_faiss_index()

    # -------------------------
    # FAISS integration
    # -------------------------
    def build_faiss_index(self):
        """Builds a FAISS index on embedding weights for nearest-neighbor search."""
        embedding_weights = (
            self.embedding.weight.detach().cpu().numpy().astype("float32")
        )
        faiss.normalize_L2(embedding_weights)
        self.faiss_index = faiss.IndexFlatIP(embedding_weights.shape[1])
        self.faiss_index.add(embedding_weights)

    # -------------------------
    # Forward pass
    # -------------------------
    def forward(
        self,
        batch_size: int,
        encoder_embedding: torch.Tensor,
        max_generation_length: int,
        use_teacher_forcing: bool = True,
        true_labels: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        device = encoder_embedding.device
        beam_size = kwargs.get("beam_size", 1)
        early_stopping = kwargs.get("early_stopping", True)
        force_max_generation = kwargs.get("force_max_generation", False)

        if use_teacher_forcing:
            # --- Teacher forcing mode ---
            input_ids = true_labels[:, :-1]
            decoder_input_embeds = self.embedding(input_ids)
            decoder_input_embeds = self.dropout(decoder_input_embeds)

            seq_length = input_ids.size(1)
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float("-inf"), device=device),
                diagonal=1,
            )
            causal_mask = (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, seq_length, seq_length)
            )
            cache_position = torch.arange(seq_length, device=device)
            encoder_attention_mask = (
                1.0 - encoder_attention_mask[:, None, None, :]
            ) * -1e9

            hidden_states = decoder_input_embeds
            for block in self.decoder_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_embedding,
                    encoder_attention_mask=encoder_attention_mask,
                    cache_position=cache_position,
                )[0]

            if self.output_projection is not None:
                hidden_states = self.final_layer_norm(hidden_states)
                logits = self.output_projection(hidden_states)
                return logits
            else:
                return hidden_states

        else:
            # --- Beam search mode ---
            assert beam_size >= 1
            encoder_embedding = (
                encoder_embedding.unsqueeze(1)
                .expand(-1, beam_size, -1, -1)
                .reshape(batch_size * beam_size, -1, encoder_embedding.shape[-1])
            )
            encoder_attention_mask = (
                encoder_attention_mask.unsqueeze(1)
                .expand(-1, beam_size, -1)
                .reshape(batch_size * beam_size, -1)
            )

            sequences = torch.full(
                (batch_size * beam_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            sequence_scores = torch.zeros(batch_size * beam_size, device=device)
            finished = torch.zeros_like(sequence_scores).bool()
            logits_sequence = torch.zeros(
                batch_size * beam_size,
                max_generation_length,
                self.vocab_size,
                device=device,
            )

            for step in range(max_generation_length):
                decoder_input_embeds = self.embedding(sequences)
                decoder_input_embeds = self.dropout(decoder_input_embeds)

                seq_length = decoder_input_embeds.size(1)
                causal_mask = (
                    torch.triu(
                        torch.full(
                            (seq_length, seq_length), float("-inf"), device=device
                        ),
                        diagonal=1,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size * beam_size, 1, seq_length, seq_length)
                )
                cache_position = torch.arange(seq_length, device=device)
                encoder_attn_mask = (
                    1.0 - encoder_attention_mask[:, None, None, :]
                ) * -1e9

                hidden_states = decoder_input_embeds
                for block in self.decoder_blocks:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        encoder_hidden_states=encoder_embedding,
                        encoder_attention_mask=encoder_attn_mask,
                        cache_position=cache_position,
                    )[0]

                # --- Output layer ---
                if self.output_projection is not None:
                    hidden_states = self.final_layer_norm(hidden_states)
                    logits = self.output_projection(hidden_states[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)
                else:
                    # --- FAISS nearest neighbor search ---
                    decoder_hidden = (
                        hidden_states[:, -1, :].detach().cpu().numpy().astype("float32")
                    )
                    faiss.normalize_L2(decoder_hidden)
                    distances, indices = self.faiss_index.search(
                        decoder_hidden, beam_size
                    )
                    indices = torch.tensor(indices, device=device, dtype=torch.long)
                    # Build pseudo-logits
                    logits = torch.full(
                        (batch_size * beam_size, self.vocab_size),
                        float("-inf"),
                        device=device,
                    )
                    for i in range(batch_size * beam_size):
                        logits[i, indices[i]] = distances[
                            i
                        ]  # use distances as pseudo-logits
                    log_probs = F.log_softmax(logits, dim=-1)

                # --- Beam selection ---
                total_scores = (sequence_scores.unsqueeze(1) + log_probs).view(
                    batch_size, beam_size * self.vocab_size
                )
                top_scores, top_indices = torch.topk(total_scores, beam_size, dim=-1)
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                # Update sequences
                sequences = sequences.view(batch_size, beam_size, -1)
                new_sequences, new_logits_sequence = [], []
                for b in range(batch_size):
                    new_seqs = sequences[b, beam_indices[b]]
                    new_tok = token_indices[b].unsqueeze(1)
                    new_seqs = torch.cat([new_seqs, new_tok], dim=-1)
                    new_sequences.append(new_seqs)
                    new_logits_sequence.append(
                        logits_sequence[b * beam_size : (b + 1) * beam_size][
                            beam_indices[b]
                        ]
                    )
                sequences = torch.stack(new_sequences, dim=0).view(
                    batch_size * beam_size, -1
                )
                logits_sequence = torch.cat(new_logits_sequence, dim=0)

                sequence_scores = top_scores.view(-1)
                finished = finished | (token_indices == self.eos_token_id).view(-1)
                if early_stopping and finished.all() and not force_max_generation:
                    break

            sequences = sequences.view(batch_size, beam_size, -1)
            sequence_scores = sequence_scores.view(batch_size, beam_size)
            logits_sequence = logits_sequence.view(
                batch_size, beam_size, -1, self.vocab_size
            )
            best_indices = torch.argmax(sequence_scores, dim=-1)
            best_logits_sequences = logits_sequence[
                torch.arange(batch_size, device=device), best_indices
            ]

            return best_logits_sequences
