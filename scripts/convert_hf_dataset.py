"""Convert ZeroAgency/ru-instruct-conversation-v1 to Lyra Chat raw JSONL.

Produces data/raw/dataset.jsonl.gz in the shape the preparation pipeline
expects: one JSON object per dialogue with keys id / turns / roles.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(["datasets"])

from datasets import load_dataset  # noqa: E402

OUT_PATH = ROOT / "data" / "raw" / "dataset.jsonl.gz"
DATASET_NAME = "ZeroAgency/ru-instruct-conversation-v1"

# Можно ограничить для быстрой проверки; None = все 82 208 примеров.
LIMIT: int | None = None

MIN_TURN_CHARS = 2
MAX_TURN_CHARS = 600
MAX_DIALOGUE_CHARS = 4000

VALID_ROLES = {"system", "user", "assistant"}


def _filter_conversation(conversations: list[dict]) -> list[dict] | None:
    """Return cleaned turns or None if the dialogue is unusable."""
    turns: list[dict] = []
    for msg in conversations:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in VALID_ROLES:
            continue
        if not (MIN_TURN_CHARS <= len(content) <= MAX_TURN_CHARS):
            return None
        turns.append({"role": role, "content": content})

    if not turns:
        return None

    # A prompt must never start with an assistant turn.
    if turns[0]["role"] == "assistant":
        return None

    # Cap total dialogue length to match DataConfig.max_dialogue_chars.
    if sum(len(t["content"]) for t in turns) > MAX_DIALOGUE_CHARS:
        return None

    return turns


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    kept = 0
    skipped = 0
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as fh:
        for index, row in enumerate(ds):
            if LIMIT is not None and index >= LIMIT:
                break
            # У ZeroAgency колонка называется "conversation" (ед. число),
            # у похожих датасетов — "conversations"; поддерживаем оба.
            conversations = row.get("conversation") or row.get("conversations")
            if not isinstance(conversations, list):
                skipped += 1
                continue
            turns = _filter_conversation(conversations)
            if turns is None:
                skipped += 1
                continue
            record = {
                "id": f"zhconv-{index}",
                "turns": [t["content"] for t in turns],
                "roles": [t["role"] for t in turns],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} dialogues to {OUT_PATH} (skipped {skipped})")
    if kept == 0:
        raise SystemExit(
            "Error: 0 dialogues kept. Check that the dataset rows have a "
            "'conversation'/'conversations' list field (see script)."
        )


if __name__ == "__main__":
    main()
