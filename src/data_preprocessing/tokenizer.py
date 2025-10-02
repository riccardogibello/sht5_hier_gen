import json
import os
from os import makedirs
import shutil
from typing import Optional, List

import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing


class HierarchicalLabelTokenizerFast:
    def __init__(
        self,
        labels_csv_file_path: Optional[str] = None,
        max_level: int = 3,
        pad_token: str = "<pad>",
        pad_token_id: int = 0,
        bos_token: str = "<s>",
        bos_token_id: int = 1,
        eos_token: str = "</s>",
        eos_token_id: int = 2,
        unk_token: str = "<unk>",
        unk_token_id: int = 3,
        tokenizer_dir: Optional[str] = None,
        per_level_chars: int = 2,
        first_level_chars: int = 1,
        cumulative: bool = False,
        label_descriptions_tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        # Classic T5 special tokens
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.label_id_descriptions = {}
        self.labels_descriptions_tokenizer = label_descriptions_tokenizer

        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
            # Load the map from label ids to descriptions
            label_id_descriptions_path = os.path.join(
                tokenizer_dir,
                "label_id_descriptions.json",
            )
            if os.path.exists(label_id_descriptions_path):
                with open(label_id_descriptions_path, "r") as f:
                    self.label_id_descriptions = json.load(f)
            else:
                print(
                    f"Warning: label_id_descriptions.json not found in {tokenizer_dir}. "
                    "Label descriptions will not be available."
                )
        else:
            vocab, self.label_id_descriptions = self._build_vocab_from_labels(
                labels_csv_file_path=labels_csv_file_path,
                max_level=max_level,
                per_level_chars=per_level_chars,
                first_level_chars=first_level_chars,
                cumulative=cumulative,
            )
            # Save the map into the tokenizer, so that it can be retrieved later
            if tokenizer_dir and not os.path.exists(tokenizer_dir):
                os.makedirs(tokenizer_dir)
            if tokenizer_dir:
                with open(
                    os.path.join(tokenizer_dir, "label_id_descriptions.json"), "w"
                ) as f:
                    json.dump(self.label_id_descriptions, f, indent=4)
            self.tokenizer = self._create_tokenizer(
                vocab,
                tokenizer_dir,
            )

        # Set special tokens for T5
        self.tokenizer.pad_token = self.pad_token
        self.tokenizer.bos_token = self.bos_token
        self.tokenizer.eos_token = self.eos_token
        self.tokenizer.unk_token = self.unk_token

    def _build_vocab_from_labels(
        self,
        labels_csv_file_path,
        max_level: int,
        per_level_chars: int,
        first_level_chars: int,
        cumulative: bool,
    ) -> tuple[List[str], dict]:
        label_descriptions = {}
        # TODO change this
        first_level_chars = 3
        df = pd.read_csv(labels_csv_file_path).dropna(
            subset=["label", "long_description"]
        )
        df["label"] = df["label"].apply(
            lambda x: (
                x if (len(x) - first_level_chars) % per_level_chars == 0 else "_" + x
            )
        )
        df["len"] = df["label"].apply(
            lambda x: (len(x) - first_level_chars) // per_level_chars + 1
        )
        df = df[df["len"] <= max_level]
        label_descriptions = {
            row["label"]: f"Label: {row["label"]}. "
            + f"Description: {row["description"]}. {row["long_description"]}"
            for _, row in df.iterrows()
        }
        # Split
        df["label"] = df["label"].apply(
            lambda x: (
                [
                    x[: first_level_chars + i * per_level_chars]
                    for i in range((len(x) - first_level_chars) // per_level_chars + 1)
                ]
                if cumulative
                else [
                    (
                        x[:first_level_chars]
                        if i == 0
                        else x[
                            first_level_chars
                            + (i - 1) * per_level_chars : first_level_chars
                            + i * per_level_chars
                        ]
                    )
                    for i in range((len(x) - first_level_chars) // per_level_chars + 1)
                ]
            )
        )
        unique = list(df["label"].explode().unique())
        # Add T5 special tokens
        vocab = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            "POP",
            "ROOT",
        ] + unique
        special_token_descriptions = {
            self.pad_token: "Padding token",
            self.bos_token: "Beginning of sequence token",
            self.eos_token: "End of sequence token",
            self.unk_token: "Unknown token",
            "POP": "Pop the last element from the stack",
            "ROOT": "Root of the hierarchy",
        }
        label_descriptions = {**special_token_descriptions, **label_descriptions}
        label_id_descriptions = {}
        for index, label in enumerate(vocab):
            if label in label_descriptions:
                old_description = label_descriptions[label]
                parent_label = label[:-per_level_chars]
                if label not in special_token_descriptions:
                    if len(parent_label) < first_level_chars:
                        parent_label = "ROOT"
                    parent_description = label_descriptions.get(
                        parent_label, "No parent description"
                    )
                    # Extract the string starting from "Description: ", if any
                    parent_subdescription = parent_description.split("Description: ")
                    if len(parent_subdescription) > 1:
                        parent_description = parent_subdescription[1]
                    else:
                        parent_description = parent_subdescription[0]
                    label_id_descriptions[index] = (
                        f"{old_description} \n\n Parent label: {parent_label}. "
                    )
                else:
                    label_id_descriptions[index] = (
                        f"Label: {label}. Description: {old_description}"
                    )
        entries = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        # Tokenize all the descriptions in a batched way for the T5 encoder, if a tokenizer is provided
        if self.labels_descriptions_tokenizer is not None:
            # Convert dict to list ordered by label id (index)
            max_idx = max(label_id_descriptions.keys()) if label_id_descriptions else -1
            descriptions_list = [
                label_id_descriptions.get(i, "") for i in range(max_idx + 1)
            ]
            # Tokenize all descriptions in a batch
            tokenization_result = self.labels_descriptions_tokenizer(
                descriptions_list,
                max_length=512,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_tensors=None,
            )
            # Build a list of dicts for each label id, containing input_ids, attention_mask, and token_type_ids if present
            label_id_descriptions = []
            for i in range(len(descriptions_list)):
                current_input_ids = tokenization_result["input_ids"][i]
                current_attention_mask = tokenization_result["attention_mask"][i]
                entries["input_ids"].append(current_input_ids)
                entries["attention_mask"].append(current_attention_mask)
                if "token_type_ids" in tokenization_result:
                    current_token_type_ids = tokenization_result["token_type_ids"][i]
                    entries["token_type_ids"].append(current_token_type_ids)

        if len(entries["token_type_ids"]) == 0:
            del entries["token_type_ids"]
        return vocab, entries

    def _create_tokenizer(
        self,
        vocab: List[str],
        tokenizer_dir: str,
    ) -> PreTrainedTokenizerFast:
        tmp_dir = os.path.join(tokenizer_dir, "tmp")
        if not os.path.exists(tmp_dir):
            makedirs(tmp_dir)
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        model = WordLevel(vocab=vocab_dict, unk_token=self.unk_token)
        tokenizer = Tokenizer(model)

        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single=self.bos_token + " $0 " + self.eos_token,
            special_tokens=[
                (self.pad_token, self.pad_token_id),
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
                (self.unk_token, self.unk_token_id),
            ],
        )
        tokenizer.enable_padding(pad_id=self.pad_token_id, pad_token=self.pad_token)
        tokenizer.add_special_tokens(
            [
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ]
        )

        tokenizer_json_path = os.path.join(tmp_dir, "tmp_hierarchical_tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
        hf_tokenizer.pad_token = self.pad_token
        hf_tokenizer.bos_token = self.bos_token
        hf_tokenizer.eos_token = self.eos_token
        hf_tokenizer.unk_token = self.unk_token
        return hf_tokenizer

    def save_pretrained(self, save_directory: str):
        tmp_dir = os.path.join(save_directory, "tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        config = {
            "pad_token": self.pad_token,
            "pad_token_id": self.pad_token_id,
            "bos_token": self.bos_token,
            "bos_token_id": self.bos_token_id,
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
            "unk_token": self.unk_token,
            "unk_token_id": self.unk_token_id,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str):
        return cls(tokenizer_dir=tokenizer_dir)

    def tokenize(self, text):
        # Do not manually add EOS if TemplateProcessing is used
        return self.tokenizer.tokenize(text)

    def __call__(self, text, *args, **kwargs):
        # Do not manually add EOS if TemplateProcessing is used
        return self.tokenizer(text, *args, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def batch_decode(self, ids_list, **kwargs):
        return self.tokenizer.batch_decode(ids_list, **kwargs)
