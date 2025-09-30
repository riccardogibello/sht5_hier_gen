from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.structured_generative_t5.config import (
    StructuredGenerativeT5Config,
)
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm
from transformers import (
    T5ForConditionalGeneration,
)


class TransformerCharDecoder(nn.Module):

    def __init__(
        self,
        config: StructuredGenerativeT5Config,
    ):
        super().__init__()

        self.config = config
        self.hidden_size = config.decoder_hidden_size
        self.vocab_size = config.decoder_vocab_size
        self.pad_token_id = config.decoder_pad_token_id
        self.bos_token_id = config.decoder_bos_token_id
        self.eos_token_id = config.decoder_eos_token_id
        self.max_length = config.decoder_max_input_length

        # === Load original pretrained T5 encoder ===
        original_model = T5ForConditionalGeneration.from_pretrained(
            config.hf_encoder_name
        )

        # === Prepare decoder config ===
        new_config = deepcopy(original_model.config)
        new_config.decoder_start_token_id = self.bos_token_id
        new_config.vocab_size = self.vocab_size
        new_config.num_decoder_layers = config.decoder_num_layers
        new_config.pad_token_id = self.pad_token_id
        new_config.eos_token_id = self.eos_token_id
        new_config.is_decoder = True
        new_config.use_cache = False

        self.dropout = nn.Dropout(new_config.dropout_rate)

        # === Input Embedding Layer ===
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id,
        )

        # === Output Projection Layer (tied with embedding) ===
        self.output_projection = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )
        self.tie_weights()

        # === Decoder Blocks ===
        self.decoder_blocks = nn.ModuleList()
        for i in range(new_config.num_decoder_layers):
            block = T5Block(
                new_config,
                layer_idx=i,
                has_relative_attention_bias=(i == 0),
            )
            self.decoder_blocks.append(block)

        # === Final Layer Norm ===
        self.final_layer_norm = T5LayerNorm(
            self.hidden_size,
            eps=new_config.layer_norm_epsilon,
        )

        # === Initialize Weights (Xavier for new modules) ===
        self._init_weights()

        # === Cleanup ===
        del original_model
        torch.cuda.empty_cache()

    def _init_weights(self):
        # Xavier initialization for embedding and projection
        nn.init.xavier_uniform_(self.embedding.weight)

        # Optional: LayerNorms are usually fine, but to be explicit:
        nn.init.ones_(self.final_layer_norm.weight)

        for block in self.decoder_blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if hasattr(module, "bias") and module.bias is not None:
                        nn.init.zeros_(module.bias)

    def tie_weights(self):
        self.output_projection.weight = self.embedding.weight

    def forward(
        self,
        batch_size: int,
        encoder_embedding: torch.Tensor,
        max_generation_length: int,
        use_teacher_forcing: bool = True,
        true_labels: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        **kwargs: dict,
    ):
        device = encoder_embedding.device
        beam_size = kwargs.get("beam_size", 1)
        early_stopping = kwargs.get("early_stopping", True)
        force_max_generation = kwargs.get("force_max_generation", False)

        # If true_labels are provided, always return logits (teacher forcing mode)
        if use_teacher_forcing:
            input_ids = true_labels[:, :-1]  # (B, T)
            decoder_input_embeds = self.embedding(input_ids)  # (B, T, H)
            decoder_input_embeds = self.dropout(decoder_input_embeds)

            seq_length = input_ids.size(1)

            # (T, T)
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float("-inf"), device=device),
                diagonal=1,
            )
            # (B, 1, T, T)
            causal_mask = (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, seq_length, seq_length)
            )

            cache_position = torch.arange(seq_length, device=device)
            # Transform the mask into the shape (B, 1, 1, S) and set 0 for positions to attend and
            # -1e9 for positions to ignore
            encoder_attention_mask = (
                1.0 - encoder_attention_mask[:, None, None, :]
            ) * -1e9
            hidden_states = decoder_input_embeds
            for block in self.decoder_blocks:
                # Get the tuple containing a list of tensors, in which the first dimension is the batch size.
                # The first tensor is of size (B, output_seq_len, hidden_size), representing the hidden states.
                # The second tensor is of size (B, num_heads, output_seq_len, output_seq_len),
                # representing the attention weights.
                # The third tensor is of size (B, num_heads, output_seq_len, input_seq_len),
                # representing the cross-attention weights.
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_embedding,
                    encoder_attention_mask=encoder_attention_mask,
                    cache_position=cache_position,
                )[0]

            hidden_states = self.final_layer_norm(hidden_states)
            logits = self.output_projection(hidden_states)

            return logits
        else:
            # === Beam Search ===
            assert beam_size >= 1, "Beam size must be at least 1 for beam search."

            # Repeat everything for beam search
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
            )  # (B*beam, 1)

            sequence_scores = torch.zeros(
                batch_size, beam_size, device=device
            )  # (B, beam)
            sequence_scores[:, 1:] = -1e9  # only keep first beam at step 0
            sequence_scores = sequence_scores.view(-1)  # (B*beam,)

            finished = torch.zeros_like(sequence_scores).bool()

            # Initialize tensor to store logits per generated token
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
                )
                causal_mask = causal_mask.expand(
                    batch_size * beam_size, 1, seq_length, seq_length
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

                hidden_states = self.final_layer_norm(hidden_states)
                logits = self.output_projection(hidden_states[:, -1, :])  # (B*beam, V)
                log_probs = F.log_softmax(logits, dim=-1)

                # Save logits at current step for each sequence
                logits_sequence[:, step, :] = logits

                total_scores = sequence_scores.unsqueeze(1) + log_probs  # (B*beam, V)
                total_scores = total_scores.view(
                    batch_size, beam_size * self.vocab_size
                )

                top_scores, top_indices = torch.topk(
                    total_scores, beam_size, dim=-1
                )  # (B, beam)
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                sequences = sequences.view(batch_size, beam_size, -1)
                new_sequences = []
                new_logits_sequence = []
                for b in range(batch_size):
                    # Gather selected beams and append new tokens
                    new_seqs = sequences[b, beam_indices[b]]  # (beam, seq_len)
                    new_tok = token_indices[b].unsqueeze(1)  # (beam, 1)
                    new_seqs = torch.cat([new_seqs, new_tok], dim=-1)
                    new_sequences.append(new_seqs)

                    # Also gather logits for selected beams
                    new_logits_sequence.append(
                        logits_sequence[b * beam_size : (b + 1) * beam_size][
                            beam_indices[b]
                        ]
                    )

                sequences = torch.stack(new_sequences, dim=0).view(
                    batch_size * beam_size, -1
                )
                logits_sequence = torch.cat(
                    new_logits_sequence, dim=0
                )  # (B*beam, step+1, V)

                sequence_scores = top_scores.view(-1)

                is_eos = token_indices == self.eos_token_id
                finished = finished | is_eos.view(-1)

                if early_stopping and finished.all() and not force_max_generation:
                    break

            sequences = sequences.view(batch_size, beam_size, -1)
            sequence_scores = sequence_scores.view(batch_size, beam_size)
            logits_sequence = logits_sequence.view(
                batch_size, beam_size, -1, self.vocab_size
            )

            best_indices = torch.argmax(sequence_scores, dim=-1)

            # Select best logits sequence for each batch element
            best_logits_sequences = logits_sequence[
                torch.arange(batch_size, device=device), best_indices
            ]  # (B, T_gen, V)

            return best_logits_sequences
