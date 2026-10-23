"""Data download, cleaning, tokenization and batching."""

from .batching import build_dataloader, collate_pretrain, make_sft_collator
from .dataset import (
    PretrainDataset,
    SFTDataset,
    load_tokenized_meta,
    tokenize_pretrain,
    tokenize_sft,
    train_tokenizer_from_corpus,
)
from .downloader import (
    RawDialogue,
    download_dataset,
    download_file,
    iter_raw_dialogues,
)
from .preprocessing import (
    SPLITS,
    assign_split,
    build_cleaned_corpus,
    clean_dialogue,
    iter_clean_dialogues,
)
from .quality import DialogueDeduplicator, DialogueQualityStats, mask_pii, normalize_text
from .roles import RoleAssignment, infer_roles

__all__ = [
    "build_dataloader",
    "collate_pretrain",
    "make_sft_collator",
    "PretrainDataset",
    "SFTDataset",
    "load_tokenized_meta",
    "tokenize_pretrain",
    "tokenize_sft",
    "train_tokenizer_from_corpus",
    "RawDialogue",
    "download_dataset",
    "download_file",
    "iter_raw_dialogues",
    "SPLITS",
    "assign_split",
    "build_cleaned_corpus",
    "clean_dialogue",
    "iter_clean_dialogues",
    "DialogueDeduplicator",
    "DialogueQualityStats",
    "mask_pii",
    "normalize_text",
    "RoleAssignment",
    "infer_roles",
]
