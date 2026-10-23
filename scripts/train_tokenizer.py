"""Train (or reload) the byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(["tokenizers", "numpy", "yaml", "tqdm"])

from conversation_llm.config import DataConfig  # noqa: E402
from conversation_llm.data.dataset import train_tokenizer_from_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the conversation tokenizer")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = DataConfig.load(root / args.data_config)
    tokenizer = train_tokenizer_from_corpus(config, root, force=args.force)
    print(
        json.dumps(
            {
                "directory": str(config.resolve(root).tokenizer_dir),
                "vocab_size": tokenizer.vocab_size,
                "special_tokens": dict(tokenizer.special_tokens),
                "fingerprint": tokenizer.fingerprint(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
