from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pytest

from conversation_llm.config import DataConfig
from conversation_llm.data.downloader import RawDialogue, iter_raw_dialogues
from conversation_llm.data.preprocessing import (
    assign_split,
    build_cleaned_corpus,
    clean_dialogue,
)
from conversation_llm.data.quality import (
    DialogueDeduplicator,
    DialogueQualityStats,
    check_turn,
    mask_pii,
    normalize_text,
    repair_mojibake,
)
from conversation_llm.data.roles import infer_roles


def make_config(**kwargs) -> DataConfig:
    base = dict(
        source_url="https://example.invalid/dataset.jsonl.gz",
        min_turns=2,
        max_turns=8,
        max_turn_chars=200,
        max_dialogue_chars=800,
        split_fractions=(0.5, 0.25, 0.25),
        seed=7,
    )
    base.update(kwargs)
    return DataConfig(**base)


def test_repair_mojibake_and_normalize() -> None:
    broken = "Привет, как дела?".encode("utf-8").decode("latin1")
    assert repair_mojibake(broken) == "Привет, как дела?"
    assert normalize_text("  Привет\u00a0 \u200b мир!!!  ") == "Привет мир!!!"
    assert normalize_text("ааааааааа") == "ааааааааа"


def test_pii_is_masked() -> None:
    text = "Пиши на test@example.com или +7 (999) 123-45-67, сайт https://x.ru"
    masked = mask_pii(text)
    assert "<почта>" in masked
    assert "<телефон>" in masked
    assert "<ссылка>" in masked


def test_turn_quality_keeps_slang_and_rejects_junk() -> None:
    assert check_turn("Здарова, чё как?")[0]
    assert check_turn("привеееет")[0]
    assert not check_turn("!!!!!")[0]
    assert not check_turn("https://spam.example")[0]
    assert not check_turn("ааааааааааа")[0]


def test_clean_dialogue_removes_bad_turns_and_masks_pii() -> None:
    config = make_config()
    stats = DialogueQualityStats()
    dedup = DialogueDeduplicator()
    record = clean_dialogue(
        RawDialogue(["Привет!", "", "Пиши на a@b.ru", "Как дела?", "!!!!"]),
        config,
        stats,
        dedup,
    )
    assert record is not None
    assert record["turns"] == ["Привет!", "Пиши на <почта>", "Как дела?"]
    assert record["roles"] == ["user", "assistant", "user"]
    assert record["roles_inferred"] is True
    assert stats.turns_dropped["empty"] == 1
    assert stats.pii.emails == 1


def test_clean_dialogue_drops_empty_duplicate_and_spam() -> None:
    config = make_config()
    stats = DialogueQualityStats()
    dedup = DialogueDeduplicator()
    assert clean_dialogue(RawDialogue(["", "  "]), config, stats, dedup) is None
    first = clean_dialogue(
        RawDialogue(["Привет, как дела?", "Нормально, а у тебя?"]), config, stats, dedup
    )
    assert first is not None
    assert dedup.check(first["turns"]) == "exact_duplicate"
    spam = clean_dialogue(
        RawDialogue(["Переходи по ссылке https://spam.ru", "Забирай промокод"]),
        config,
        stats,
        dedup,
    )
    assert spam is None
    assert stats.dropped["too_few_valid_turns"] >= 1


def test_roles_are_inferred_and_confidence_bounded() -> None:
    assignment = infer_roles(["Привет! Как дела?", "Нормально, а у тебя?"])
    assert assignment.inferred
    assert assignment.roles == ["user", "assistant"]
    assert 0.5 <= assignment.confidence <= 0.85
    explicit = infer_roles(["Привет", "Здравствуй"], provided=["assistant", "user"])
    assert not explicit.inferred
    assert explicit.roles == ["assistant", "user"]
    with pytest.raises(ValueError):
        infer_roles(["Привет"], provided=["unknown"])
    with pytest.raises(ValueError):
        infer_roles([], provided=[])


def test_split_assignment_is_deterministic() -> None:
    config = make_config()
    names = {assign_split("some dialogue", config.split_fractions, config.seed) for _ in range(5)}
    assert names == {assign_split("some dialogue", config.split_fractions, config.seed)}
    assert assign_split("some dialogue", config.split_fractions, config.seed) in {"train", "val", "test"}


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_downloader_reads_gzip_jsonl_json_and_zip(tmp_path: Path) -> None:
    rows = [
        {"sample": ["Привет", "Здравствуй"]},
        {"sample": ["Как дела?", "Хорошо"]},
        {"bad": "ignored"},
    ]
    gz = tmp_path / "data.jsonl.gz"
    _write_gzip_jsonl(gz, rows)
    assert len(list(iter_raw_dialogues(gz))) == 2

    plain = tmp_path / "data.jsonl"
    plain.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    assert len(list(iter_raw_dialogues(plain))) == 2

    as_json = tmp_path / "data.json"
    as_json.write_text(
        json.dumps([{"messages": [{"role": "user", "content": "Привет"}]}], ensure_ascii=False),
        encoding="utf-8",
    )
    parsed = list(iter_raw_dialogues(as_json))
    assert parsed[0].turns == ["Привет"]
    assert parsed[0].roles == ["user"]

    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data.jsonl", json.dumps(rows[0], ensure_ascii=False))
    assert len(list(iter_raw_dialogues(archive))) == 1

    with pytest.raises(FileNotFoundError):
        list(iter_raw_dialogues(tmp_path / "missing.jsonl"))


def test_build_cleaned_corpus_splits_and_caches(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw.jsonl"
    dialogues = []
    for index in range(12):
        dialogues.append({"sample": [f"Привет, диалог {index}", f"Ответ на диалог {index}"]})
    dialogues.append({"sample": ["Привет, диалог 0", "Ответ на диалог 0"]})  # duplicate
    dialogues.append({"sample": []})
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dialogues), encoding="utf-8"
    )
    config = make_config(raw_path=str(raw))
    report = build_cleaned_corpus(config, tmp_path, download=False)
    assert report["quality"]["kept"] == 12
    assert report["quality"]["dropped"]["exact_duplicate"] == 1
    assert sum(report["splits"][name]["dialogues"] for name in ("train", "val", "test")) == 12
    lines = []
    for split in ("train", "val", "test"):
        path = tmp_path / "artifacts" / "data" / "cleaned" / f"{split}.jsonl"
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    assert len(lines) == 12
    assert all(json.loads(line)["roles"][0] == "user" for line in lines)
    second = build_cleaned_corpus(config, tmp_path, download=False)
    assert second["signature"] == report["signature"]
