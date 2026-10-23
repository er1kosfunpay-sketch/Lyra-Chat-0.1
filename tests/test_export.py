from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conversation_llm.config import GenerationConfig
from conversation_llm.inference.generator import ChatGenerator, load_export_directory

ROOT = Path(__file__).resolve().parents[1]


def test_export_is_self_contained_and_loadable(
    tokenizer, tiny_checkpoint: Path, tmp_path: Path
) -> None:
    tokenizer_dir = tmp_path / "tokenizer-export"
    tokenizer.save(tokenizer_dir)
    output = tmp_path / "exports"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_model.py"),
            "--checkpoint",
            str(tiny_checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output-dir",
            str(output),
            "--name",
            "tiny-export",
            "--root",
            str(ROOT),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    export_dir = output / "tiny-export"
    for filename in (
        "model.safetensors",
        "model_config.json",
        "generation.yaml",
        "export_info.json",
        "checksums.sha256",
        "verification.json",
        "README.md",
        "tokenizer/tokenizer.json",
    ):
        assert (export_dir / filename).exists(), filename
    assert (output / "tiny-export.zip").exists()
    info = json.loads((export_dir / "export_info.json").read_text(encoding="utf-8"))
    assert info["parameters"] > 0
    verification = json.loads((export_dir / "verification.json").read_text(encoding="utf-8"))
    assert verification["verified"] is True

    model, loaded_tokenizer, config, loaded_info = load_export_directory(
        export_dir, device="cpu"
    )
    assert loaded_info["step"] == 3
    generator = ChatGenerator(
        model,
        loaded_tokenizer,
        config=GenerationConfig(do_sample=False, max_new_tokens=2),
        device="cpu",
    )
    result = generator.chat("Привет")
    assert result.generated_tokens <= 2
