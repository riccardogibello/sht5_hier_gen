from torch import nn
import torch

from src.architectures.structured_generative_t5.config import (
    StructuredGenerativeT5Config,
)
from src.utils.constants import DECODER_DROPOUT_FIELD, DECODER_DROPOUT_VALUE


class LstmDecoder(nn.Module):

    def __init__(
        self,
        config: StructuredGenerativeT5Config,
    ):
        super().__init__()

        self.config = config

        self.lstm_embedding_matrix = nn.Embedding(
            num_embeddings=config.decoder_vocab_size,
            embedding_dim=config.decoder_input_size,
        )
        self.lstm_embedding_matrix.weight = nn.init.xavier_uniform_(
            self.lstm_embedding_matrix.weight
        )

        self.nn_linear = nn.Linear(
            in_features=config.decoder_hidden_size,
            out_features=config.decoder_vocab_size,
        )
        self.nn_linear.weight = nn.init.xavier_uniform_(
            self.nn_linear.weight,
        )

        self.lstm_model = nn.LSTM(
            input_size=config.decoder_input_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.decoder_num_layers,
            batch_first=True,
            dropout=config.get_field(
                DECODER_DROPOUT_FIELD,
                DECODER_DROPOUT_VALUE,
            ),
        )

        self.dropout = nn.Dropout(
            config.get_field(
                DECODER_DROPOUT_FIELD,
                DECODER_DROPOUT_VALUE,
            )
        )
        self.relu = nn.ReLU()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(
        self,
        batch_size: int,
        encoder_embedding: torch.LongTensor,
        max_generation_length: int,
        use_teacher_forcing: bool,
        true_labels: torch.LongTensor = None,
        **kwargs: dict,
    ):
        assert (
            not use_teacher_forcing or true_labels is not None
        ), "If teacher forcing is used, true_labels must be provided."
        # Set the initial LSTM output to the BOS token embedding
        lstm_input = (
            self.lstm_embedding_matrix(
                torch.tensor(
                    self.config.decoder_bos_token_id,
                    device=self.device,
                )
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        lstm_logits_list = []

        num_layers = self.config.decoder_num_layers
        hidden_size = self.config.decoder_hidden_size
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)
        if encoder_embedding.size(-1) != hidden_size:
            encoder_embedding = self.encoder_proj(encoder_embedding)
        h_0[0] = encoder_embedding
        lstm_state = (h_0, c_0)

        # For each timestep that can be generated
        for t in range(1, max_generation_length + 1):
            # If teacher forcing is enabled, use as input the previous true label
            if use_teacher_forcing:
                lstm_input = self.lstm_embedding_matrix(true_labels[:, t - 1])

            # Reshape the input to (batch_size, 1, lstm_hidden_size)
            lstm_input = lstm_input.view(
                lstm_input.size(0), 1, self.config.decoder_input_size
            )
            # Move the input to the device specified in the config
            lstm_input = lstm_input.to(self.device)

            # Pass the input through the LSTM model
            lstm_output, lstm_state = self.lstm_model(lstm_input, lstm_state)
            # Get the logits for the next token
            lstm_logits = self.nn_linear(lstm_output)
            # Apply dropout and activation function
            lstm_logits = self.relu(lstm_logits)
            lstm_logits = self.dropout(lstm_logits)
            # Append the output to the list
            lstm_logits_list.append(lstm_logits)
            # Get the next input token
            lstm_input = lstm_logits.argmax(dim=-1)
            # Reshape the input to (batch_size, 1, lstm_hidden_size)
            lstm_input = self.lstm_embedding_matrix(lstm_input).unsqueeze(1)

        logits = torch.cat(lstm_logits_list, dim=1)

        return logits

    def old_forward(
        self,
        batch_size: int,
        encoder_embedding: torch.Tensor,  # shape: (batch_size, enc_seq_len, enc_hidden_size)
        max_generation_length: int,
        use_teacher_forcing: bool,
        true_labels: torch.LongTensor = None,
    ):
        assert (
            not use_teacher_forcing or true_labels is not None
        ), "If teacher forcing is used, true_labels must be provided."

        # Initialize the first LSTM input with the BOS token embedding
        lstm_input = (
            self.lstm_embedding_matrix(
                torch.tensor(self.config.decoder_bos_token_id, device=self.device)
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Initialize LSTM hidden and cell states
        num_layers = self.config.decoder_num_layers
        input_size = self.config.decoder_input_size
        hidden_size = self.config.decoder_hidden_size
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)

        lstm_state = (h_0, c_0)
        lstm_logits_list = []

        # Project encoder_embedding if needed to match decoder hidden size
        if encoder_embedding.size(-1) != hidden_size:
            encoder_embedding = self.encoder_proj(
                encoder_embedding
            )  # Add this projection in __init__

        # Initialize the first hidden state with the encoder CLS token, which is the first token in the sequence
        h_0[0] = encoder_embedding[:, 0, :]  # (batch_size, hidden_size)

        # Compute attention keys and values (precompute for efficiency)
        # Assuming dot-product attention
        encoder_keys = encoder_embedding  # (batch_size, enc_seq_len, hidden_size)
        encoder_values = encoder_embedding

        for t in range(1, max_generation_length + 1):
            if use_teacher_forcing:
                lstm_input = self.lstm_embedding_matrix(true_labels[:, t - 1])

            lstm_input = lstm_input.view(batch_size, 1, input_size).to(self.device)

            # LSTM forward
            lstm_output, lstm_state = self.lstm_model(
                lstm_input, lstm_state
            )  # lstm_output: (batch_size, 1, hidden_size)
            dec_hidden = lstm_output.squeeze(1)  # (batch_size, hidden_size)

            # Attention score computation: (batch_size, enc_seq_len)
            attn_scores = torch.bmm(
                encoder_keys,
                dec_hidden.unsqueeze(-1),
            ).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Context vector: (batch_size, hidden_size)
            context_vector = torch.bmm(
                attn_weights.unsqueeze(1), encoder_values
            ).squeeze(1)

            # Combine context and LSTM output
            # Combine context and LSTM output
            combined_output = torch.cat([dec_hidden, context_vector], dim=-1)
            combined_output = self.attn_combine(combined_output)
            combined_output = self.relu(combined_output)  # Add non-linearity
            combined_output = self.dropout(combined_output)  # Add regularization

            lstm_output = combined_output.unsqueeze(1)
            # Get the logits for the next token
            lstm_logits = self.nn_linear(lstm_output)
            lstm_logits_list.append(lstm_logits)
            # Get the next input token
            lstm_input = lstm_logits.argmax(dim=-1)
            lstm_input = self.lstm_embedding_matrix(lstm_input).unsqueeze(1)

        logits = torch.cat(lstm_logits_list, dim=1)

        return logits
