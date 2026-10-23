"""Export a self-contained, verified model bundle and ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(["torch", "tokenizers", "numpy", "yaml", "tqdm", "safetensors"])

import torch  # noqa: E402

from conversation_llm.config import DataConfig, GenerationConfig, ModelConfig  # noqa: E402
from conversation_llm.inference.generator import load_model_from_checkpoint  # noqa: E402
from conversation_llm.tokenizer import ConversationTokenizer  # noqa: E402
from conversation_llm.training.checkpoint import read_checkpoint_metadata  # noqa: E402
from conversation_llm.utils.logging import setup_logger  # noqa: E402

LOGGER = setup_logger("conversation_llm.export")
EXPORT_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{_sha256(path)}  {path.relative_to(directory).as_posix()}")
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_readme(info: dict, metrics: dict) -> str:
    steps = metrics.get("steps")
    smoke_note = ""
    if isinstance(steps, int) and steps < 100:
        smoke_note = (
            "> **ВНИМАНИЕ:** это smoke-экспорт для проверки цепочки "
            f"({steps} шагов обучения), а не полноценно обученная модель. "
            "Ответы будут бессвязными.\n\n"
        )
    return f"""# {info['name']}

{smoke_note}Локальная русскоязычная разговорная модель, обученная с нуля
(decoder-only Transformer, byte-level BPE). Внешние LLM/API не используются.

## Состав

- `model.safetensors` — веса модели;
- `model_config.json` — архитектура;
- `tokenizer/` — обученный токенизатор (стабильные id специальных токенов);
- `generation.yaml` — параметры генерации по умолчанию;
- `export_info.json`, `training_metrics.json`, `dataset_info.json`;
- `checksums.sha256`.

## Запуск

Из корня проекта:

```bash
python scripts/chat.py --export-dir "{info['export_dir']}"
```

Одна реплика без интерактивного режима:

```bash
python scripts/chat.py --export-dir "{info['export_dir']}" --message "Привет! Как дела?"
```

Inference работает на CPU (медленнее) и CUDA. Требуются только PyTorch, tokenizers,
numpy, PyYAML, safetensors (см. `requirements.txt`).

## Данные и ограничения

Источник: {info['dataset']['name']} (лицензия {info['dataset']['license']}).
Роли говорящих в исходном корпусе отсутствуют и были восстановлены эвристикой
(чередование, начиная с пользователя). Модель небольшая (~{info['parameters']/1e6:.1f}M
параметров), поэтому ошибки, повторы и фактические неточности ожидаемы.
Метрики: train loss {metrics.get('final_train_loss')}, validation loss
{metrics.get('final_val_loss')}, best val loss {metrics.get('best_val_loss')}.
"""


def _verify_export(export_dir: Path, expected_parameters: int) -> dict:
    code = (
        "from conversation_llm.config import GenerationConfig\n"
        "from conversation_llm.inference.generator import load_export_directory, ChatGenerator\n"
        f"model, tok, cfg, info = load_export_directory({str(export_dir)!r}, device='cpu')\n"
        "assert tok.vocab_size == cfg.vocab_size\n"
        "gen = ChatGenerator(model, tok, device='cpu')\n"
        "prompt = [tok.bos_id, tok.user_id] + tok.encode_text('Привет') + [tok.turn_end_id, tok.assistant_id]\n"
        "result = gen.generate(prompt, GenerationConfig(do_sample=False, max_new_tokens=2))\n"
        "print('EXPORT_OK', model.num_parameters(), result.stop_reason)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as tmp:
        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp,
            env=env,
            text=True,
            capture_output=True,
            timeout=900,
        )
    if process.returncode != 0:
        raise RuntimeError(
            "Export verification failed in a fresh process.\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return {
        "verified": True,
        "process_cwd": "temporary directory",
        "stdout": process.stdout.strip(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "parameters": expected_parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained model bundle")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (required)")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--generation-config", default=None)
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--name", default=None)
    parser.add_argument("--evaluation-report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required: pass the path to a trained model checkpoint")

    root = Path(args.root).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if args.tokenizer:
        tokenizer_dir = Path(args.tokenizer)
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = root / tokenizer_dir
    else:
        tokenizer_dir = DataConfig.load(root / args.data_config).resolve(root).tokenizer_dir

    model, tokenizer, model_config, info = load_model_from_checkpoint(
        checkpoint, tokenizer_dir, device="cpu"
    )
    metadata = read_checkpoint_metadata(checkpoint)
    stage = metadata.get("training_config", {}).get("stage", "sft")
    step = metadata.get("step")
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    name = args.name or f"conversation-llm-{stage}-step{step or 0}"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    export_dir = output_dir / name
    if export_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Export directory exists: {export_dir}. Use --overwrite.")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    # Weights
    state = {
        key: value.detach().cpu().clone().contiguous()
        for key, value in model.state_dict().items()
    }
    from safetensors.torch import save_file

    save_file(state, str(export_dir / "model.safetensors"))
    (export_dir / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Tokenizer
    tokenizer_export = export_dir / "tokenizer"
    tokenizer_export.mkdir()
    for filename in ("tokenizer.json", "special_tokens.json", "tokenizer_config.json"):
        source = tokenizer_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file missing: {source}")
        shutil.copy2(source, tokenizer_export / filename)
    loaded_tokenizer = ConversationTokenizer.load(tokenizer_export)
    if loaded_tokenizer.fingerprint() != tokenizer.fingerprint():
        raise RuntimeError("Copied tokenizer does not round-trip with the same fingerprint")

    generation = (
        GenerationConfig.load(root / args.generation_config)
        if args.generation_config
        else GenerationConfig()
    )
    generation.save(export_dir / "generation.yaml")

    training_metrics = _read_json(run_dir / "training_report.json")
    if not training_metrics:
        training_metrics = {
            "stage": stage,
            "steps": step,
            "tokens_seen": metadata.get("tokens_seen"),
            "best_val_loss": metadata.get("best_val_loss"),
            "note": "training_report.json was not found next to the checkpoint",
        }
    (export_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_report = _read_json(root / "reports" / "data_report.json")
    tokenization_report = _read_json(root / "reports" / "tokenization_report.json")
    dataset_info = {
        "name": data_report.get("source", {}).get("name", "Den4ikAI/russian_dialogues_2"),
        "license": data_report.get("source", {}).get("license", "MIT"),
        "url": data_report.get("source", {}).get("url"),
        "raw_dialogues_read": data_report.get("source", {}).get("raw_dialogues_read"),
        "raw_turns": data_report.get("source", {}).get("raw_turns"),
        "kept_dialogues": data_report.get("quality", {}).get("kept"),
        "kept_turns": data_report.get("quality", {}).get("turns_kept"),
        "split_sizes": data_report.get("splits"),
        "tokenization": {
            "pretrain": tokenization_report.get("pretrain", {}).get("splits"),
            "sft": tokenization_report.get("sft", {}).get("splits"),
        },
        "role_annotation": "inferred by alternation; not ground truth",
        "preprocessing": {
            "dropped": data_report.get("quality", {}).get("dropped"),
            "turns_dropped": data_report.get("quality", {}).get("turns_dropped"),
            "pii_masked": data_report.get("quality", {}).get("pii"),
            "truncated_turns": data_report.get("quality", {}).get("truncated_turns"),
        },
        "raw_dataset_included": False,
    }
    (export_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.evaluation_report:
        report_path = Path(args.evaluation_report)
        if not report_path.is_absolute():
            report_path = root / report_path
        if report_path.is_dir():
            for filename in ("evaluation_report.json", "evaluation_report.md", "generations.jsonl"):
                source = report_path / filename
                if source.exists():
                    shutil.copy2(source, export_dir / filename)
        elif report_path.exists():
            shutil.copy2(report_path, export_dir / report_path.name)

    export_info = {
        "export_version": EXPORT_VERSION,
        "name": name,
        "export_dir": str(export_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "step": step,
        "smoke": bool(isinstance(step, int) and step < 100),
        "tokens_seen": metadata.get("tokens_seen"),
        "best_val_loss": metadata.get("best_val_loss"),
        "parameters": model.num_parameters(),
        "model_config": model_config.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "source_run_dir": str(run_dir),
        "dataset": {"name": dataset_info["name"], "license": dataset_info["license"]},
        "format": "safetensors",
    }
    (export_dir / "export_info.json").write_text(
        json.dumps(export_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (export_dir / "README.md").write_text(
        _export_readme(export_info, training_metrics), encoding="utf-8"
    )

    if not args.skip_verify:
        LOGGER.info("Verifying export in a fresh Python process ...")
        verification = _verify_export(export_dir, model.num_parameters())
        (export_dir / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Export verified: %s", verification["stdout"])

    _write_checksums(export_dir)
    archive_path = None
    if not args.no_zip:
        archive_base = output_dir / name
        archive_path = shutil.make_archive(
            str(archive_base), "zip", root_dir=export_dir.parent, base_dir=export_dir.name
        )
        LOGGER.info("Archive created: %s (%.1f MB)", archive_path, Path(archive_path).stat().st_size / 1e6)

    print(
        json.dumps(
            {
                "export_dir": str(export_dir),
                "archive": archive_path,
                "parameters": model.num_parameters(),
                "step": step,
                "weights_mb": round((export_dir / "model.safetensors").stat().st_size / 1e6, 1),
                "verified": not args.skip_verify,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

