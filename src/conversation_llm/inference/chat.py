"""Terminal chat REPL for the locally trained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import GenerationConfig
from ..utils.device import resolve_device
from ..utils.environment import ensure_dependencies
from ..utils.logging import setup_logger
from .generator import (
    ChatGenerator,
    load_export_directory,
    load_model_from_checkpoint,
)

DEFAULT_SYSTEM_PROMPT = (
    "Ты — полезный русскоязычный ассистент. "
    "Отвечай по существу, вежливо и кратко, на русском языке. "
    "Не ролепLAY, не придумывай истории, не продолжай чужие диалоги — "
    "просто ответь на вопрос пользователя."
)


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a locally trained conversation model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export-dir", help="Directory produced by scripts/export_model.py")
    source.add_argument("--checkpoint", help="Path to best.pt / last.pt / step_*.pt")
    parser.add_argument("--tokenizer", help="Tokenizer directory (required with --checkpoint)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--message", help="Run a single message and exit (non-interactive)")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Системная инструкция модели (роль ассистента)",
    )
    return parser


def _apply_overrides(config: GenerationConfig, args: argparse.Namespace) -> GenerationConfig:
    payload = config.to_dict()
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    for key, value in mapping.items():
        if value is not None:
            payload[key] = value
    if args.greedy:
        payload["do_sample"] = False
    return GenerationConfig(**payload)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdout()
    ensure_dependencies(["torch", "tokenizers", "numpy", "yaml", "tqdm", "safetensors"])
    args = build_parser().parse_args(argv)
    logger = setup_logger("conversation_llm.chat")

    if args.export_dir:
        model, tokenizer, model_config, info = load_export_directory(
            args.export_dir, device=args.device
        )
        gen_config = GenerationConfig.load(Path(args.export_dir) / "generation.yaml")
    else:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required together with --checkpoint")
        model, tokenizer, model_config, info = load_model_from_checkpoint(
            args.checkpoint, args.tokenizer, device=args.device
        )
        gen_config = GenerationConfig()
    gen_config = _apply_overrides(gen_config, args)
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    generator = ChatGenerator(
        model,
        tokenizer,
        config=gen_config,
        device=resolve_device(args.device),
        system_prompt=system_prompt,
    )

    logger.info(
        "Loaded model: %s parameters, vocab=%d, device=%s, checkpoint step=%s",
        f"{model.num_parameters():,}",
        tokenizer.vocab_size,
        generator.device,
        info.get("step"),
    )
    print("=" * 66)
    print(" Разговорная модель (локальный инференс, без внешних API)")
    print(" Команды: /reset  /history  /length N  /temp X  /topk N  /topp X  /help  /quit")
    print("=" * 66)

    if args.message is not None:
        result = generator.chat(args.message)
        if result.text:
            print(result.text)
        else:
            print(f"[пустой ответ; stop={result.stop_reason}]")
        return 0

    while True:
        try:
            user = input("\nВы: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nВыход.")
            break
        if not user:
            continue
        if user.startswith("/"):
            command, _, value = user.partition(" ")
            command = command.lower()
            if command in {"/quit", "/exit"}:
                break
            if command == "/reset":
                generator.reset()
                print("[история очищена]")
                continue
            if command == "/history":
                print(generator.history_as_text())
                continue
            if command == "/help":
                print(
                    "/reset — очистить историю\n"
                    "/history — показать историю\n"
                    "/length N — максимум новых токенов\n"
                    "/temp X — температура (0 = greedy)\n"
                    "/topk N — top-k (0 = выключить)\n"
                    "/topp X — top-p\n"
                    "/quit — выход"
                )
                continue
            try:
                if command == "/length":
                    generator.config.max_new_tokens = max(1, int(value))
                elif command == "/temp":
                    generator.config.temperature = max(0.0, float(value))
                elif command == "/topk":
                    generator.config.top_k = max(0, int(value))
                elif command == "/topp":
                    generator.config.top_p = min(1.0, max(0.01, float(value)))
                else:
                    print("[неизвестная команда; /help]")
                    continue
            except ValueError:
                print("[некорректное значение]")
                continue
            print(f"[{command} = {value}]")
            continue

        try:
            result = generator.chat(user)
        except (RuntimeError, ValueError) as exc:
            print(f"[ошибка генерации: {exc}]")
            continue
        if result.text:
            print(f"Модель: {result.text}")
        else:
            print(f"Модель: [пустой ответ; stop={result.stop_reason}]")
        print(
            f"        ({result.generated_tokens} токенов, {result.stop_reason})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
