import os
import shutil

from src.architectures.checkpointing import load_checkpoint
from src.architectures.structured_generative_t5.config import (
    BaselineT5Config,
)
from src.architectures.baseline_t5.model.baseline_t5_model import BaselineT5
from src.utils.constants import CONFIG_FILE_NAME

import torch

from src.visualization.attention_visualization import (
    build_decoder_cross_attention_matrix,
    build_decoder_self_attention_matrix,
    plot_interactive_attention_heatmap,
)

INPUTS = [
    """
        Term: Ophthalmic curette, reusable -- 
        Definition: A hand-held, manual, ophthalmic surgical instrument typically with a fenestrated, spoon-shaped or ring-like tip which can be either sharp or blunt, intended to be used to obtain or remove eye tissue through a scraping action. It is typically made of metal. This is a reusable device.
    """,
    """
        Term: Ossicular prosthesis, total -- Definition: A sterile device intended to be implanted for the total functional reconstruction of the ossicular chain (mallues, incus, and stapes bones) to facilitate the conduction of sound waves from the tympanic membrane to the inner ear. It is designed to treat conductive hearing loss from traumatic or surgical injury, otosclerosis, congenital fixation of the stapes, or chronic middle ear disease. The device is typically in the form of a tube made of polymers.
    """,
    """
        Term: External manual massager, home-use -- Definition: A manual (non-powered) device specifically designed to apply external massage in the home. It is typically held in the hand of the user and drawn/rolled across the area of the body/muscles to be treated; it may also be designed to be placed on the floor for foot massage. The massaging components, e.g., roller heads or pads, may be interchangeable with others of different size and shape. This is a reusable device.
    """,
]

EXPERIMENT_NAME = "emdn_gmdn"
MODEL_FOLDER_NAME = "t5_baseline_t5_small_2pl_cum_True"

DATA_FOLDER_PATH = os.path.join(
    ".",
    "data",
)
EXPERIMENT_FOLDER_PATH = os.path.join(
    DATA_FOLDER_PATH,
    EXPERIMENT_NAME,
)
MODELS_FOLDER_PATH = os.path.join(
    EXPERIMENT_FOLDER_PATH,
    "models",
)
MODEL_FOLDER_PATH = os.path.join(
    MODELS_FOLDER_PATH,
    MODEL_FOLDER_NAME,
)
EXAMPLES_FOLDER_PATH = os.path.join(
    MODEL_FOLDER_PATH,
    "examples",
)
if os.path.exists(EXAMPLES_FOLDER_PATH):
    shutil.rmtree(EXAMPLES_FOLDER_PATH)
os.makedirs(EXAMPLES_FOLDER_PATH, exist_ok=True)

if __name__ == "__main__":
    config: BaselineT5Config = BaselineT5Config.from_pretrained(
        load_directory=MODEL_FOLDER_PATH,
        file_name=CONFIG_FILE_NAME,
    )
    model: BaselineT5 = BaselineT5(
        local_folder_path=MODEL_FOLDER_PATH,
        hf_model_name=config.hf_encoder_name,
        config=config,
    )
    load_checkpoint(
        model=model,
        model_folder_path=MODEL_FOLDER_PATH,
        load_best=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Clean all the inputs, keeping only alphanumeric characters, symbols, and spaces
    INPUTS = [
        " ".join(
            word
            for word in input_text.split()
            if word.isalnum()
            or word.isspace()
            or any(char in word for char in ".,-;:!?()[]{}'\"/\\&%$#@*+=~`|<>")
        )
        for input_text in INPUTS
    ]

    rows = []
    columns = [
        "Input Text",
        "Output",
    ]
    for i, input_text in enumerate(INPUTS):
        print(input_text)
        # Tokenize the input text with the model's tokenizer
        input_ids = model.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=config.max_source_length,
            truncation=True,
        ).to(device)
        # Get the token IDs and attention mask
        attention_mask = (input_ids != model.tokenizer.pad_token_id).long()

        (
            generated_ids,
            all_decoder_self_attns,
            all_cross_attns,
            input_tokens,
            output_tokens,
        ) = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.max_target_length,
            use_cache=True,
        )
        output_folder_path = os.path.join(
            EXAMPLES_FOLDER_PATH,
            f"example_{i + 1}",
        )
        os.makedirs(output_folder_path, exist_ok=True)

        build_decoder_self_attention_matrix(
            output_tokens=output_tokens[0],
            all_decoder_self_attns=all_decoder_self_attns,
            output_folder_path=output_folder_path,
        )

        cross_attn_matrix, output_tokens, input_tokens = (
            build_decoder_cross_attention_matrix(
                output_tokens=output_tokens[0],
                input_tokens=input_tokens[0],
                all_decoder_cross_attns=all_cross_attns,
                output_folder_path=output_folder_path,
            )
        )
        plot_interactive_attention_heatmap(
            cross_attention_matrix=cross_attn_matrix,
            output_tokens=output_tokens,
            merged_input_tokens=input_tokens,
        )

        rows.append(
            [
                input_text,
                model.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                ),
            ]
        )
