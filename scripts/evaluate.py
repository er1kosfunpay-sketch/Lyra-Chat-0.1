"""Evaluate validation/test loss and fixed dialogue scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(["torch", "tokenizers", "numpy", "yaml", "tqdm", "safetensors"])

import torch  # noqa: E402

from conversation_llm.config import DataConfig, GenerationConfig  # noqa: E402
from conversation_llm.evaluation.evaluator import (  # noqa: E402
    build_eval_loader,
    evaluate_loss,
    generate_scenarios,
    save_evaluation,
    text_quality_metrics,
)
from conversation_llm.inference.generator import (  # noqa: E402
    ChatGenerator,
    load_export_directory,
    load_model_from_checkpoint,
)
from conversation_llm.training.checkpoint import read_checkpoint_metadata  # noqa: E402
from conversation_llm.utils.device import resolve_device  # noqa: E402
from conversation_llm.utils.logging import setup_logger  # noqa: E402

LOGGER = setup_logger("conversation_llm.evaluate")


def _history_summary(history: list[float]) -> dict[str, float | int] | None:
    if not history:
        return None
    return {
        "count": len(history),
        "first": history[0],
        "last": history[-1],
        "min": min(history),
        "max": max(history),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained conversation model")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--data-config", default="configs/data.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--export-dir")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", choices=["pretrain", "sft"], default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_config = DataConfig.load(root / args.data_config)
    device = resolve_device(args.device)
    started = time.time()

    info: dict = {}
    if args.export_dir:
        export_dir = Path(args.export_dir)
        if not export_dir.is_absolute():
            export_dir = root / export_dir
        model, tokenizer, model_config, info = load_export_directory(export_dir, device=device)
        gen_config = GenerationConfig.load(export_dir / "generation.yaml")
        stage = args.stage or info.get("stage", "sft")
        checkpoint_label = str(export_dir)
        run_dir = Path(info.get("source_run_dir", "")) if info.get("source_run_dir") else None
    else:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if args.tokenizer:
            tokenizer_dir = Path(args.tokenizer)
            if not tokenizer_dir.is_absolute():
                tokenizer_dir = root / tokenizer_dir
        else:
            tokenizer_dir = data_config.resolve(root).tokenizer_dir
        model, tokenizer, model_config, info = load_model_from_checkpoint(
            checkpoint, tokenizer_dir, device=device
        )
        payload_meta = read_checkpoint_metadata(checkpoint)
        info = {**info, **payload_meta}
        stage = args.stage or payload_meta.get("training_config", {}).get("stage", "sft")
        gen_config = GenerationConfig()
        checkpoint_label = str(checkpoint)
        run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent

    generation = GenerationConfig.from_dict(
        {
            **gen_config.to_dict(),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": not args.greedy,
        }
    )

    data_root = root / "artifacts" / "data" / "tokenized" / stage
    loss_results: dict[str, dict | None] = {}
    dataset_info: dict[str, dict] = {}
    for split in ("val", "test"):
        meta_path = data_root / split / "meta.json"
        if not meta_path.exists():
            LOGGER.warning("Tokenized %s split not found: %s", split, meta_path)
            loss_results[split] = None
            continue
        loader, meta = build_eval_loader(
            stage,
            data_root,
            split,
            batch_size=args.batch_size,
            pad_id=tokenizer.pad_id,
            seed=args.seed,
        )
        LOGGER.info("Evaluating %s (%s examples, %s tokens)", split, meta.get("examples"), meta.get("tokens"))
        loss_results[split] = evaluate_loss(
            model, loader, device, max_batches=args.max_batches, amp="auto"
        )
        dataset_info[split] = {
            "examples": meta.get("examples"),
            "tokens": meta.get("tokens"),
            "response_tokens": meta.get("response_tokens"),
            "max_len": meta.get("max_len"),
        }

    train_loss = None
    training_val_loss = None
    history = None
    if run_dir and (Path(run_dir) / "training_report.json").exists():
        training_report = json.loads(
            (Path(run_dir) / "training_report.json").read_text(encoding="utf-8")
        )
        history = {
            "train_loss": _history_summary(training_report.get("train_loss_history", [])),
            "val_loss": _history_summary(training_report.get("val_loss_history", [])),
            "steps": training_report.get("steps"),
            "tokens_seen": training_report.get("tokens_seen"),
        }
        train_loss = training_report.get("final_train_loss")
        training_val_loss = training_report.get("final_val_loss")

    generator = ChatGenerator(
        model, tokenizer, config=generation, device=device
    )
    if args.skip_generation:
        scenarios: list[dict] = []
        metrics = {}
    else:
        LOGGER.info("Generating fixed evaluation scenarios")
        scenarios = generate_scenarios(generator, generation, seed=args.seed)
        metrics = text_quality_metrics(scenarios)

    report = {
        "checkpoint": checkpoint_label,
        "step": info.get("step"),
        "tokens_seen": info.get("tokens_seen"),
        "parameters": model.num_parameters(),
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.describe(),
        "stage": stage,
        "device": str(device),
        "loss": loss_results,
        "dataset": dataset_info,
        "train_loss": train_loss,
        "training_val_loss": training_val_loss,
        "training_history": history,
        "generation_config": generation.to_dict(),
        "metrics": metrics,
        "scenarios": scenarios,
        "elapsed_sec": round(time.time() - started, 2),
        "notes": [
            "Roles in the source corpus are inferred by alternation, not annotated.",
            "Test split is never used for training or checkpoint selection.",
        ],
    }
    output = Path(args.output) if args.output else (root / "reports" / "evaluation")
    if not output.is_absolute():
        output = root / output
    json_path, md_path = save_evaluation(output, report, scenarios)
    print(
        json.dumps(
            {
                "report": str(json_path),
                "markdown": str(md_path),
                "val_loss": (loss_results.get("val") or {}).get("loss"),
                "test_loss": (loss_results.get("test") or {}).get("loss"),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
