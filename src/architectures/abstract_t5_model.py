import gc
import os
from typing import Iterator, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm
from src.architectures.structured_generative_t5.config import (
    TrainingConfig,
)
from src.utils.constants import OUTPUT_DATA_FOLDER_NAME


class AbstractT5Model(nn.Module):

    def __init__(
        self,
        local_folder_path: str,
    ):
        nn.Module.__init__(self)

        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path, exist_ok=True)
        self.model_path = local_folder_path

        # Setup device and move model to device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_device(self):
        """
        Get the device on which the model is currently loaded.

        :return: The device (e.g., 'cuda' or 'cpu').
        """
        return self._device

    def step(
        self,
        data_generator: Iterator[Tuple],
        log_message_prefix: str = "train",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        previous_steps: int = 0,
        training_config: Optional[TrainingConfig] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Run the train, test, or validation epoch over the given data.

        :param data_generator: The data generator to fetch all the provided data.
        :param log_message_prefix: The mode of operation, to be prefixed in the log messages.
        :param optimizer: (Optional) The optimizer to be used for updating the model parameters.
        :param scheduler: (Optional) The learning rate scheduler to be used for updating the learning rate.
        :param previous_steps: The number of steps already performed before the current epoch.
        :param training_config: The training configuration containing the hyperparameters.
        :param kwargs: Additional keyword arguments, which can include:
            - after_warmup_callback: A callback function to be called after the warmup steps.
            - store_mappings_callback: A callback function to store the mappings of source texts, true labels, and predicted labels.

        :return: A tuple containing the average loss and variance over the batch.
        """
        total_loss = 0
        n_accum = 0
        accumulated_loss = 0
        is_training = optimizer is not None
        batch_n = 0
        progress_bar = tqdm(data_generator, desc=f"{log_message_prefix}")
        accumulated_losses = []
        total_batches = 0

        after_warmup_callback = kwargs.get("after_warmup_callback", None)
        store_mappings_callback = kwargs.get("store_mappings_callback", None)

        file_folder = os.path.join(
            self.model_path,
            OUTPUT_DATA_FOLDER_NAME,
        )
        file_name = f"{log_message_prefix}.csv"
        is_test = log_message_prefix == "test"
        if is_test:
            test_cleaned_file_name = f"{log_message_prefix}_cleaned.csv"
            test_cleaned_file_path = os.path.join(file_folder, test_cleaned_file_name)
            if os.path.exists(test_cleaned_file_path):
                return 0.0, 0.0
            file_path = os.path.join(file_folder, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        for i, batch in enumerate(progress_bar):
            batch_n += 1
            total_batches += 1
            global_step = (i // training_config.accum_iter) + previous_steps

            is_callback_given = after_warmup_callback is not None
            is_last_warmup_step = global_step == training_config.warmup_steps
            if is_last_warmup_step and is_callback_given:
                after_warmup_callback()

            # Get the input tensors from the batch, along with its attention mask, token type ids, and labels
            source_input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            token_type_ids = batch[3]
            source_input_ids = source_input_ids.to(self._device)
            labels = labels.to(self._device)

            loss, _, predicted_token_ids = self.forward(
                input_ids=source_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_teacher_forcing=not is_test,
                training_config=training_config,
            )

            # Accumulate the scaled loss
            accumulated_loss += loss
            if (i + 1) % training_config.accum_iter == 0:
                # Average the accumulated loss over the number of accumulation steps
                accumulated_loss = accumulated_loss / training_config.accum_iter
                # Store the accumulated loss in the list
                accumulated_losses.append(accumulated_loss.to("cpu").item())
                if is_training:
                    # Perform the backward pass to compute the gradients
                    accumulated_loss.backward()
                    # If the optimizer is provided
                    if optimizer is not None:
                        # If the scaler is provided to perform the mixed precision training
                        # Perform the optimization step without scaling the gradients
                        optimizer.step()
                        # If a learning rate scheduler is provided
                        if scheduler is not None:
                            # Update the learning rate
                            scheduler.step()
                        # Reset the gradients to zero to allow the freeing of the memory
                        optimizer.zero_grad()
                # Reset the accumulated loss
                accumulated_loss = 0
                # Update the number of accumulation steps
                n_accum += 1
            if (is_test or i % 100 == 0) and store_mappings_callback is not None:
                store_mappings_callback(
                    file_folder=file_folder,
                    file_name=f"{log_message_prefix}.csv",
                    src_texts=source_input_ids,
                    true_label_tensor=labels,
                    predicted_logits=predicted_token_ids,
                )
            # Keep track of the unscaled loss and the total number of tokens to finally compute the average loss value.
            total_loss += loss
            # Update the progress bar
            lr = (
                optimizer.param_groups[0]["lr"]
                if optimizer is not None
                else float("nan")
            )
            progress_bar.set_postfix(
                accumulation_step=n_accum,
                loss=(total_loss / total_batches).item(),
                lr=lr,
            )

        avg_loss = total_loss / total_batches
        avg_loss = avg_loss
        # Compute the variance of the accumulated losses
        loss_variance = torch.var(torch.tensor(accumulated_losses))
        loss_variance = loss_variance.to("cpu")

        # Return the average loss and variance over the batch
        return avg_loss.to("cpu").detach().numpy(), loss_variance.item()
