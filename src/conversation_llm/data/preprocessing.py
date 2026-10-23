"""Cleaning, role reconstruction, de-duplication and deterministic splitting."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..config import DataConfig
from ..utils.logging import setup_logger
from .downloader import RawDialogue, download_file, iter_raw_dialogues
from .manifest import atomic_write_json, file_fingerprint, manifest_matches, stable_hash
from .quality import (
    DialogueDeduplicator,
    DialogueQualityStats,
    PiiCounts,
    check_turn,
    letter_count,
    mask_pii,
    ngram_repetition,
    normalize_text,
    summarize_lengths,
    truncate_text,
    word_fingerprint,
)
from .roles import infer_roles

LOGGER = setup_logger("conversation_llm.data.preprocessing")
SPLITS = ("train", "val", "test")
CLEANING_VERSION = 2
_PLACEHOLDER_ONLY = {
    "<ссылка>",
    "<почта>",
    "<телефон>",
    "<упоминание>",
    "<номер>",
}


def assign_split(dialogue_key: str, fractions: tuple[float, float, float], seed: int) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{dialogue_key}".encode("utf-8"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") / float(2**64)
    cumulative = 0.0
    for name, fraction in zip(SPLITS, fractions):
        cumulative += fraction
        if bucket < cumulative:
            return name
    return SPLITS[-1]


def clean_dialogue(
    raw: RawDialogue,
    config: DataConfig,
    stats: DialogueQualityStats,
    deduplicator: DialogueDeduplicator,
) -> dict[str, Any] | None:
    """Clean one dialogue; return a serializable record or None (and count why)."""
    stats.seen += 1
    stats.turns_seen += len(raw.turns)
    kept_turns: list[str] = []
    kept_roles: list[str] | None = [] if raw.roles is not None else None

    for index, turn in enumerate(raw.turns):
        text = normalize_text(turn)
        if not text:
            stats.turns_dropped["empty"] += 1
            continue
        ok, reason = check_turn(text, config.min_letters_per_turn)
        if not ok:
            stats.turns_dropped[reason] += 1
            continue
        local_pii = PiiCounts()
        text = mask_pii(text, local_pii)
        stats.note_pii(local_pii)
        if text.strip() in _PLACEHOLDER_ONLY:
            stats.turns_dropped["placeholder_only"] += 1
            continue
        text, truncated = truncate_text(text, config.max_turn_chars)
        if truncated:
            stats.truncated_turns += 1
        kept_turns.append(text)
        if kept_roles is not None and raw.roles is not None:
            role = raw.roles[index]
            kept_roles.append(role)

    if len(kept_turns) < config.min_turns:
        stats.drop("too_few_valid_turns")
        return None
    if sum(letter_count(turn) for turn in kept_turns) < config.min_letters_per_turn * config.min_turns:
        stats.drop("too_little_text")
        return None

    # Cap extremely long threads.  Keep the most recent turns so that replies
    # still have local context.  For inferred roles, keep an even count so the
    # first kept turn remains a human turn.
    if len(kept_turns) > config.max_turns:
        start = len(kept_turns) - config.max_turns
        if raw.roles is None and start % 2 == 1:
            start += 1
        kept_turns = kept_turns[start:]
        if kept_roles is not None:
            kept_roles = kept_roles[start:]
        stats.truncated_dialogues += 1

    total_chars = sum(len(t) for t in kept_turns)
    while total_chars > config.max_dialogue_chars and len(kept_turns) > config.min_turns:
        removed = kept_turns.pop()
        total_chars -= len(removed)
        if kept_roles is not None:
            kept_roles.pop()
        stats.truncated_dialogues += 1

    repetition = ngram_repetition(kept_turns)
    if repetition > 0.5:
        stats.drop("repetitive_fragments")
        return None

    duplicate = deduplicator.check(kept_turns)
    if duplicate is not None:
        stats.drop(duplicate)
        return None

    try:
        assignment = infer_roles(kept_turns, provided=kept_roles)
    except ValueError:
        stats.drop("invalid_roles")
        return None
    if assignment.inferred:
        stats.roles_inferred += 1
        stats.role_confidences.append(assignment.confidence)
    else:
        stats.roles_provided += 1

    stats.kept += 1
    stats.turns_kept += len(kept_turns)
    stats.turn_lengths.extend(len(t) for t in kept_turns)
    stats.dialogue_lengths.append(total_chars)
    stats.turns_per_dialogue.append(len(kept_turns))
    stats.repetition_scores.append(repetition)

    dialogue_id = raw.source_id or stable_hash(kept_turns, length=20)
    return {
        "id": str(dialogue_id),
        "turns": kept_turns,
        "roles": assignment.roles,
        "roles_inferred": assignment.inferred,
        "confidence": round(assignment.confidence, 4),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    q = report["quality"]
    splits = report["splits"]
    lines = [
        "# Отчёт о подготовке данных",
        "",
        f"Источник: `{report['source']['name']}` (лицензия {report['source']['license']}).",
        f"Сырой файл: `{report['source']['raw_file']}`, {report['source']['raw_mb']} МБ.",
        "",
        "## Итоговые сплиты",
        "",
        "| Сплит | Диалогов | Реплик | Символов |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in SPLITS:
        lines.append(
            f"| {name} | {splits[name]['dialogues']:,} | "
            f"{splits[name]['turns']:,} | {splits[name]['chars']:,} |"
        )
    lines += [
        "",
        f"Всего просмотрено диалогов: **{q['seen']:,}**, оставлено: **{q['kept']:,}**.",
        f"Реплик просмотрено: {q['turns_seen']:,}, оставлено: {q['turns_kept']:,}.",
        f"Удалено реплик по причинам: {json.dumps(q['turns_dropped'], ensure_ascii=False)}.",
        f"Удалено диалогов по причинам: {json.dumps(q['dropped'], ensure_ascii=False)}.",
        "",
        "## Длины",
        "",
        f"Длина реплики (символы): {json.dumps(q['turn_length_summary'], ensure_ascii=False)}",
        f"Длина диалога (символы): {json.dumps(q['dialogue_length_summary'], ensure_ascii=False)}",
        f"Реплик в диалоге: {json.dumps(q['turns_per_dialogue_summary'], ensure_ascii=False)}",
        "",
        "## Роли и качество",
        "",
        f"Роли восстановлены эвристикой: {q['roles_inferred']:,} диалогов; "
        f"явные метки: {q['roles_provided']:,}.",
        f"Средняя эвристическая уверенность: {q['mean_role_confidence']}.",
        f"Доля вопросов в чётных репликах (потенциальный пользователь): "
        f"{report['source']['question_rate_even']:.3f}; "
        f"в нечётных (потенциальный ассистент): {report['source']['question_rate_odd']:.3f}.",
        "**Важно:** исходный корпус не содержит идентификаторов говорящих. Роли назначены "
        "чередованием, начиная с пользователя. Это не ground truth, а восстанавливаемая "
        "разметка; её качество ограничено и отражено в confidence.",
        "",
        f"PII-маскирование: {json.dumps(q['pii'], ensure_ascii=False)}.",
        f"Обрезано длинных реплик: {q['truncated_turns']:,}; диалогов: {q['truncated_dialogues']:,}.",
        f"Средняя доля повторов 4-грамм: {q['mean_repetition']}.",
        f"Near-duplicate фильтр: {'включён' if report['near_duplicate_filter'] else 'выключен'} "
        "(отпечаток мешка слов, не MinHash).",
        "",
        "Сплит выполнен по стабильному хэшу диалога до нарезки на окна, поэтому "
        "реплики одного диалога не попадают в разные сплиты.",
    ]
    return "\n".join(lines) + "\n"


def build_cleaned_corpus(
    config: DataConfig,
    root: str | Path,
    force: bool = False,
    limit: int | None = None,
    download: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    resolved = config.resolve(root)
    resolved.cleaned_dir.mkdir(parents=True, exist_ok=True)
    resolved.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved.cleaned_dir / "manifest.json"
    effective_limit = limit if limit is not None else config.max_dialogues
    signature = stable_hash(
        {
            "version": CLEANING_VERSION,
            "data": config.to_dict(),
            "limit": effective_limit,
            "raw": file_fingerprint(resolved.raw_path) if resolved.raw_path.exists() else None,
        }
    )
    if not force and manifest_matches(manifest_path, signature):
        split_files = [resolved.cleaned_dir / f"{name}.jsonl" for name in SPLITS]
        if all(path.exists() for path in split_files):
            LOGGER.info("Cleaned corpus cache is valid: %s", resolved.cleaned_dir)
            return json.loads(manifest_path.read_text(encoding="utf-8"))

    if not resolved.raw_path.exists():
        if not download:
            raise FileNotFoundError(f"Raw dataset not found: {resolved.raw_path}")
        download_file(config.source_url, resolved.raw_path)

    stats = DialogueQualityStats()
    deduplicator = DialogueDeduplicator(
        near_duplicates=config.near_duplicate_filter,
        min_words_for_near=max(6, config.min_turns * 2),
    )
    split_counts = {
        name: {"dialogues": 0, "turns": 0, "chars": 0} for name in SPLITS
    }
    raw_turns = 0
    raw_question_even = 0
    raw_turns_even = 0
    raw_question_odd = 0
    raw_turns_odd = 0

    temp_paths = {name: resolved.cleaned_dir / f".{name}.jsonl.tmp" for name in SPLITS}
    handles = {
        name: temp_paths[name].open("w", encoding="utf-8", newline="\n") for name in SPLITS
    }
    try:
        for index, raw in enumerate(iter_raw_dialogues(resolved.raw_path)):
            if effective_limit is not None and index >= effective_limit:
                break
            for position, turn in enumerate(raw.turns):
                raw_turns += 1
                if position % 2 == 0:
                    raw_turns_even += 1
                    raw_question_even += int(turn.strip().endswith("?"))
                else:
                    raw_turns_odd += 1
                    raw_question_odd += int(turn.strip().endswith("?"))
            record = clean_dialogue(raw, config, stats, deduplicator)
            if record is None:
                continue
            key = word_fingerprint(record["turns"])
            split = assign_split(key, config.split_fractions, config.seed)
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            split_counts[split]["dialogues"] += 1
            split_counts[split]["turns"] += len(record["turns"])
            split_counts[split]["chars"] += sum(len(t) for t in record["turns"])
            if (index + 1) % 100_000 == 0:
                LOGGER.info(
                    "Processed %d raw dialogues, kept %d. Latest split sizes: %s",
                    index + 1,
                    stats.kept,
                    {name: split_counts[name]["dialogues"] for name in SPLITS},
                )
    finally:
        for fh in handles.values():
            fh.close()

    for name in SPLITS:
        os.replace(temp_paths[name], resolved.cleaned_dir / f"{name}.jsonl")

    report: dict[str, Any] = {
        "source": {
            "name": config.source_name,
            "url": config.source_url,
            "license": config.source_license,
            "raw_file": str(resolved.raw_path),
            "raw_mb": round(resolved.raw_path.stat().st_size / 1e6, 2),
            "raw_dialogues_read": stats.seen,
            "raw_turns": raw_turns,
            "question_rate_even": raw_question_even / max(raw_turns_even, 1),
            "question_rate_odd": raw_question_odd / max(raw_turns_odd, 1),
            "speaker_labels": "absent; roles inferred by alternation",
        },
        "seed": config.seed,
        "split_fractions": list(config.split_fractions),
        "near_duplicate_filter": config.near_duplicate_filter,
        "max_turns": config.max_turns,
        "max_turn_chars": config.max_turn_chars,
        "max_dialogue_chars": config.max_dialogue_chars,
        "limit": effective_limit,
        "splits": split_counts,
        "quality": {
            "seen": stats.seen,
            "kept": stats.kept,
            "turns_seen": stats.turns_seen,
            "turns_kept": stats.turns_kept,
            "dropped": dict(stats.dropped),
            "turns_dropped": dict(stats.turns_dropped),
            "truncated_turns": stats.truncated_turns,
            "truncated_dialogues": stats.truncated_dialogues,
            "pii": stats.pii.to_dict(),
            "turn_length_summary": summarize_lengths(stats.turn_lengths),
            "dialogue_length_summary": summarize_lengths(stats.dialogue_lengths),
            "turns_per_dialogue_summary": summarize_lengths(stats.turns_per_dialogue),
            "mean_repetition": (
                round(sum(stats.repetition_scores) / len(stats.repetition_scores), 4)
                if stats.repetition_scores
                else 0.0
            ),
            "roles_inferred": stats.roles_inferred,
            "roles_provided": stats.roles_provided,
            "mean_role_confidence": (
                round(sum(stats.role_confidences) / len(stats.role_confidences), 4)
                if stats.role_confidences
                else None
            ),
        },
        "signature": signature,
    }
    atomic_write_json(manifest_path, report)
    atomic_write_json(resolved.reports_dir / "data_report.json", report)
    (resolved.reports_dir / "data_report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    LOGGER.info(
        "Cleaning done: kept %d/%d dialogues; splits=%s",
        stats.kept,
        stats.seen,
        {name: split_counts[name]["dialogues"] for name in SPLITS},
    )
    return report


def iter_clean_dialogues(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
