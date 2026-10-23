"""Download, clean, train the tokenizer and tokenize the corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(
    [
        "torch",
        "tokenizers",
        "numpy",
        "yaml",
        "tqdm",
        "safetensors",
        "requests",
        "datasets",
        "huggingface_hub",
    ]
)

from conversation_llm.data.dataset import (  # noqa: E402
    tokenize_pretrain,
    tokenize_sft,
    train_tokenizer_from_corpus,
)
from conversation_llm.data.manifest import atomic_write_json, read_json  # noqa: E402
from conversation_llm.data.preprocessing import build_cleaned_corpus  # noqa: E402
from conversation_llm.config import DataConfig, ModelConfig  # noqa: E402
from conversation_llm.utils.logging import setup_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Russian dialogue dataset")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Use only N raw dialogues")
    parser.add_argument("--force", action="store_true", help="Re-run every cached stage")
    parser.add_argument("--skip-tokenization", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_config = DataConfig.load(root / args.data_config)
    model_config = ModelConfig.load(root / args.model_config)
    logger = setup_logger("conversation_llm.prepare", root / "reports" / "prepare_data.log")

    logger.info("Step 1/3: download and clean")
    report = build_cleaned_corpus(
        data_config, root, force=args.force, limit=args.limit, download=True
    )
    tokenizer = train_tokenizer_from_corpus(data_config, root, force=args.force)
    logger.info("Tokenizer ready: vocab=%d", tokenizer.vocab_size)
    if tokenizer.vocab_size != model_config.vocab_size:
        logger.warning(
            "Tokenizer vocab_size=%d differs from model.yaml (%d). "
            "Training will use the tokenizer size, but update configs/model.yaml "
            "to keep exported configs self-consistent.",
            tokenizer.vocab_size,
            model_config.vocab_size,
        )

    tokenization = None
    if not args.skip_tokenization:
        logger.info("Step 2/3: tokenize pretraining streams")
        pretrain = tokenize_pretrain(
            data_config,
            root,
            tokenizer,
            block_size=model_config.block_size,
            force=args.force,
        )
        logger.info("Step 3/3: tokenize response-only SFT examples")
        sft = tokenize_sft(
            data_config,
            root,
            tokenizer,
            block_size=model_config.block_size,
            force=args.force,
        )
        tokenization = {"pretrain": pretrain, "sft": sft}
        report["tokenization"] = tokenization
        atomic_write_json(
            data_config.resolve(root).cleaned_dir / "manifest.json", report
        )
        atomic_write_json(
            data_config.resolve(root).reports_dir / "data_report.json", report
        )

    summary = {
        "dialogues_kept": report["quality"]["kept"],
        "turns_kept": report["quality"]["turns_kept"],
        "splits": report["splits"],
        "tokenizer_vocab": tokenizer.vocab_size,
        "tokenization": tokenization,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
