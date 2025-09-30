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

        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        else:
            vocab = self._build_vocab_from_labels(
                labels_csv_file_path,
                max_level,
                per_level_chars,
            )
            self.tokenizer = self._create_tokenizer(vocab, tokenizer_dir)

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
    ):
        df = pd.read_csv(labels_csv_file_path).dropna(subset=["label"])
        df["label"] = df["label"].apply(
            lambda x: x if len(x) % per_level_chars == 0 else "_" + x
        )
        df["len"] = df["label"].apply(len) / per_level_chars
        df = df[df["len"] <= max_level]
        df["label"] = df["label"].apply(
            lambda x: [
                x[i : i + per_level_chars] for i in range(0, len(x), per_level_chars)
            ]
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
        return vocab

    def _create_tokenizer(
        self, vocab: List[str], tokenizer_dir: str
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
