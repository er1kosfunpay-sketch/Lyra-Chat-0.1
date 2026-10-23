"""Quantitative and qualitative evaluation of a trained checkpoint."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from ..config import GenerationConfig
from ..data.batching import build_dataloader, make_sft_collator
from ..inference.generator import ChatGenerator
from ..model import GPT
from ..training.metrics import perplexity
from ..utils.device import amp_settings
from .prompts import SCENARIOS, EvaluationScenario

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ROLE_BLEED_RE = re.compile(
    r"(?i)\b(?:вы|пользователь|user|assistant|ассистент|модель)\s*:"
)


@torch.no_grad()
def evaluate_loss(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    amp: str = "auto",
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    enabled, dtype, _ = amp_settings(device, amp)
    total_loss = 0.0
    total_targets = 0
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        targets = int((labels != -100).sum().item())
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if output.loss is None or not math.isfinite(float(output.loss)):
            raise FloatingPointError("Non-finite loss during evaluation")
        total_loss += float(output.loss) * targets
        total_targets += targets
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if was_training:
        model.train()
    if total_targets == 0:
        return {"loss": None, "perplexity": None, "batches": batches, "targets": 0}
    loss = total_loss / total_targets
    return {
        "loss": loss,
        "perplexity": perplexity(loss),
        "batches": batches,
        "targets": total_targets,
    }


def _repetition_rate(text: str, n: int = 4) -> float:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _distinct_n(texts: Sequence[str], n: int) -> float | None:
    words = [word for text in texts for word in _WORD_RE.findall(text.lower())]
    if len(words) < n:
        return None
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def generate_scenarios(
    generator: ChatGenerator,
    config: GenerationConfig,
    seed: int = 1234,
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> list[dict[str, Any]]:
    scenarios = scenarios or SCENARIOS
    results: list[dict[str, Any]] = []
    total_tokens = 0
    started = time.time()
    for scenario_index, scenario in enumerate(scenarios):
        torch.manual_seed(seed + scenario_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + scenario_index)
        generator.reset()
        turns: list[dict[str, Any]] = []
        for message in scenario.messages:
            result = generator.chat(message, config=config)
            turns.append({"user": message, **result.to_dict()})
            total_tokens += result.generated_tokens
        results.append(
            {"name": scenario.name, "tags": scenario.tags, "turns": turns}
        )
    elapsed = time.time() - started
    for scenario in results:
        scenario["tokens_per_sec"] = (
            round(
                sum(turn["generated_tokens"] for turn in scenario["turns"])
                / max(elapsed / max(len(results), 1), 1e-9),
                1,
            )
        )
    return results


def text_quality_metrics(scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    responses = [
        turn["text"]
        for scenario in scenarios
        for turn in scenario["turns"]
        if isinstance(turn.get("text"), str)
    ]
    nonempty = [text for text in responses if text.strip()]
    all_turns = [turn for scenario in scenarios for turn in scenario["turns"]]
    stops = [turn.get("stop_reason", "") for turn in all_turns]
    lengths = [len(text) for text in nonempty]
    repetitions = [_repetition_rate(text) for text in nonempty]
    adjacent_duplicates = 0
    comparisons = 0
    for scenario in scenarios:
        texts = [turn["text"].strip() for turn in scenario["turns"]]
        for left, right in zip(texts, texts[1:]):
            comparisons += 1
            if left and left == right:
                adjacent_duplicates += 1
    bleed = sum(1 for text in responses if _ROLE_BLEED_RE.search(text))
    return {
        "responses": len(responses),
        "nonempty": len(nonempty),
        "empty_rate": round(1.0 - len(nonempty) / max(len(responses), 1), 4),
        "stopped_rate": round(
            sum(1 for stop in stops if stop in {"<|turn_end|>", "<|eos|>"})
            / max(len(stops), 1),
            4,
        ),
        "max_tokens_rate": round(
            sum(1 for stop in stops if stop == "max_new_tokens") / max(len(stops), 1), 4
        ),
        "mean_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "distinct_1": _distinct_n(nonempty, 1),
        "distinct_2": _distinct_n(nonempty, 2),
        "mean_repetition_4gram": round(sum(repetitions) / len(repetitions), 4)
        if repetitions
        else 0.0,
        "adjacent_duplicate_rate": round(adjacent_duplicates / max(comparisons, 1), 4),
        "role_bleed_rate": round(bleed / max(len(responses), 1), 4),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Отчёт об оценке модели",
        "",
        f"Чекпойнт: `{report['checkpoint']}`",
        f"Шаг обучения: {report.get('step')}; обработано токенов: {report.get('tokens_seen')}",
        f"Параметров: {report.get('parameters'):,}" if report.get("parameters") else "Параметров: n/a",
        f"Устройство: `{report['device']}`",
        "",
        "## Loss / perplexity",
        "",
        "| Сплит | Loss | Perplexity | Target-токены | Бачей |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("val", "test"):
        entry = report["loss"].get(split)
        if entry:
            ppl = entry["perplexity"]
            lines.append(
                f"| {split} | {entry['loss']:.4f} | "
                f"{'n/a' if ppl is None else f'{ppl:.2f}'} | "
                f"{entry['targets']:,} | {entry['batches']} |"
            )
    lines += [
        "",
        f"Train loss (последний из training_report): {report.get('train_loss')}",
        f"Validation loss (последний из training_report): {report.get('training_val_loss')}",
        "",
        "## Метрики генерации",
        "",
        "| Метрика | Значение |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Примеры диалогов",
        "",
    ]
    for scenario in report["scenarios"]:
        lines.append(f"### {scenario['name']} ({', '.join(scenario['tags'])})")
        lines.append("")
        for turn in scenario["turns"]:
            lines.append(f"**Вы:** {turn['user']}")
            lines.append("")
            text = turn["text"] if turn["text"] else f"[пустой ответ, stop={turn['stop_reason']}]"
            lines.append(f"**Модель:** {text}")
            lines.append("")
            lines.append(
                f"<sub>stop={turn['stop_reason']}, tokens={turn['generated_tokens']}, "
                f"prompt_tokens={turn['prompt_tokens']}</sub>"
            )
            lines.append("")
    lines += [
        "## Ограничения",
        "",
        "Это небольшая модель, обученная на разговорном корпусе с восстановленными "
        "эвристикой ролями. Она не обязана отвечать фактически верно и не является "
        "аналогом GPT-2 по интеллектуальным способностям. Примеры выше не отфильтрованы.",
    ]
    return "\n".join(lines) + "\n"


def save_evaluation(
    output_dir: str | Path,
    report: dict[str, Any],
    generations: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as fh:
        for scenario in generations:
            for turn in scenario["turns"]:
                fh.write(
                    json.dumps(
                        {"scenario": scenario["name"], "tags": scenario["tags"], **turn},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return json_path, md_path


def build_eval_loader(
    stage: str,
    data_dir: str | Path,
    split: str,
    batch_size: int = 4,
    num_workers: int = 0,
    pad_id: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, dict[str, Any]]:
    from ..data.dataset import PretrainDataset, SFTDataset

    data_dir = Path(data_dir)
    if stage == "pretrain":
        meta = json.loads((data_dir / split / "meta.json").read_text(encoding="utf-8"))
        dataset = PretrainDataset(data_dir / split / "tokens.bin", meta["block_size"])
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            pin_memory=False,
        )
    elif stage == "sft":
        meta = json.loads((data_dir / split / "meta.json").read_text(encoding="utf-8"))
        dataset = SFTDataset(data_dir / split)
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=make_sft_collator(pad_id),
            num_workers=num_workers,
            seed=seed,
            pin_memory=False,
        )
    else:
        raise ValueError("stage must be pretrain or sft")
    return loader, meta
