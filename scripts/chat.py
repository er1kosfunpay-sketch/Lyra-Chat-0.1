"""Launch the terminal chat."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(["torch", "tokenizers", "numpy", "yaml", "tqdm", "safetensors"])

from conversation_llm.inference.chat import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
