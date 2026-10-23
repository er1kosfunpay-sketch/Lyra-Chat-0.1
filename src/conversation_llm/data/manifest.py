"""Small cache manifest helpers shared by data preparation and tokenization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def stable_hash(payload: Any, length: int = 16) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "name": path.name}


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def manifest_matches(path: str | Path, expected: dict[str, Any]) -> bool:
    payload = read_json(path, default={})
    return bool(payload) and payload.get("signature") == expected
