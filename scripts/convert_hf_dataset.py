"""Convert ZeroAgency/ru-instruct-conversation-v1 to Lyra Chat raw JSONL.

Produces data/raw/dataset.jsonl.gz in the shape the preparation pipeline
expects: one JSON object per dialogue with keys id / turns / roles.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
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
MAX_TURN_CHARS = 3000  # слишком длинные реплики (чаще всего испорченные записи); сам pipeline дополнительно урежет до 600

VALID_ROLES = {"system", "user", "assistant"}


def _filter_conversation(conversations: list[dict]) -> tuple[list[dict] | None, str]:
    """Return cleaned turns + reason, or (None, reason) if unusable.

    Filter is deliberately gentle: the project pipeline (cleaning) already
    truncates long turns/dialogues; here we only drop hopeless dialogues.
    """
    if not all(isinstance(m, dict) for m in conversations):
        return None, "bad_message_type"
    turns: list[dict] = []
    for msg in conversations:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in VALID_ROLES:
            return None, f"role_{role or 'empty'}"
        if not (MIN_TURN_CHARS <= len(content) <= MAX_TURN_CHARS):
            return None, "turn_len"
        turns.append({"role": role, "content": content})

    # A prompt must never start with an assistant turn.
    if turns[0]["role"] == "assistant":
        return None, "assistant_first"

    # SFT learns assistant turns only; require at least one, with a user
    # turn somewhere before it (otherwise no valid SFT example exists).
    if not any(t["role"] == "assistant" for t in turns):
        return None, "no_assistant"
    for i, t in enumerate(turns):
        if t["role"] == "assistant" and any(t2["role"] == "user" for t2 in turns[:i]):
            break
    else:
        return None, "assistant_without_user"

    return turns, "ok"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    kept = 0
    skipped = 0
    reasons: Counter[str] = Counter()
    first_row_keys: set[str] | None = None
    has_conversation_field = False
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as fh:
        for index, row in enumerate(ds):
            if LIMIT is not None and index >= LIMIT:
                break
            if first_row_keys is None and isinstance(row, dict):
                first_row_keys = set(row.keys())
            # У ZeroAgency колонка называется "conversation" (ед. число),
            # у похожих датасетов — "conversations"; поддерживаем оба.
            conversations = row.get("conversation") or row.get("conversations")
            if not isinstance(conversations, list):
                skipped += 1
                reasons["no_conversation_field"] += 1
                continue
            has_conversation_field = True
            turns, reason = _filter_conversation(conversations)
            if turns is None:
                skipped += 1
                reasons[reason] += 1
                continue
            record = {
                "id": f"zhconv-{index}",
                "turns": [t["content"] for t in turns],
                "roles": [t["role"] for t in turns],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} dialogues to {OUT_PATH} (skipped {skipped})")
    print("Skip reasons:", dict(reasons.most_common()))
    if first_row_keys and not has_conversation_field:
        print("Row keys sample:", sorted(first_row_keys))
    if kept == 0:
        raise SystemExit(
            "Error: 0 dialogues kept — see skip reasons above."
        )
    if kept == 0:
        raise SystemExit(
            "Error: 0 dialogues kept. Check that the dataset rows have a "
            "'conversation'/'conversations' list field (see script)."
        )


if __name__ == "__main__":
    main()
