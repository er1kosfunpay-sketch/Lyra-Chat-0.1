"""Downloading and reading the raw Russian dialogue corpus.

The primary source is a gzipped JSONL file on the Hugging Face Hub.  The
reader also accepts plain JSONL/JSON/ZIP files so the preparation code can be
reused with other corpora.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import requests

from ..config import DataConfig
from ..utils.logging import setup_logger

LOGGER = setup_logger("conversation_llm.data.downloader")


@dataclass
class RawDialogue:
    """A sequence of messages plus optional speaker labels from the source.

    `roles` is None when the source does not provide trustworthy speaker
    labels.  The preparation pipeline never fabricates labels in that case;
    it records that roles were inferred heuristically.
    """

    turns: list[str]
    roles: list[str] | None = None
    source_id: str | None = None


def _content_length(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        if response.ok:
            value = response.headers.get("content-length")
            return int(value) if value else None
    except (requests.RequestException, ValueError):
        return None
    return None


def download_file(
    url: str,
    destination: str | Path,
    retries: int = 8,
    timeout: tuple[int, int] = (30, 120),
) -> Path:
    """Download with HTTP range resume and bounded retries."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _content_length(url)
    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        LOGGER.info("Using existing download: %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
        return destination

    part = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        if expected is not None and done >= expected:
            break
        headers = {"Accept-Encoding": "identity"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code == 416 and expected is not None and done >= expected:
                    break
                response.raise_for_status()
                mode = "ab" if done and response.status_code == 206 else "wb"
                if mode == "wb":
                    done = 0
                with part.open(mode) as fh:
                    for chunk in response.iter_content(1024 * 512):
                        if chunk:
                            fh.write(chunk)
            if expected is None or part.stat().st_size >= expected:
                break
            LOGGER.warning(
                "Connection ended early (%d/%s bytes), resuming", part.stat().st_size, expected
            )
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            LOGGER.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(min(2 * attempt, 10))

    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"Could not download {url}: {last_error}")
    if expected is not None and part.stat().st_size != expected:
        raise RuntimeError(
            f"Incomplete download: {part.stat().st_size} of {expected} bytes. "
            "Re-run the command to resume."
        )
    os.replace(part, destination)
    LOGGER.info("Downloaded %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def _dialogue_from_object(obj: Any) -> RawDialogue | None:
    """Accept the common JSON dialogue shapes and normalize them."""
    turns: Sequence[Any] | None = None
    roles: list[str] | None = None
    source_id: str | None = None

    if isinstance(obj, dict):
        source_id = str(obj.get("id")) if obj.get("id") is not None else None
        for key in ("sample", "turns", "dialogue", "messages", "conversations", "text"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                turns = obj[key]
                break
    elif isinstance(obj, (list, tuple)):
        turns = obj
    if turns is None:
        return None

    texts: list[str] = []
    explicit_roles: list[str] = []
    has_roles = True
    for turn in turns:
        if isinstance(turn, str):
            texts.append(turn)
            has_roles = False
        elif isinstance(turn, dict):
            value = turn.get("content", turn.get("text", turn.get("value")))
            role = turn.get("role", turn.get("from", turn.get("author")))
            if value is None:
                has_roles = False
                continue
            texts.append(str(value))
            if role is not None:
                explicit_roles.append(str(role).lower())
            else:
                has_roles = False
        else:
            texts.append(str(turn))
            has_roles = False
    if not texts:
        return None
    if has_roles and len(explicit_roles) == len(texts):
        roles = explicit_roles
    return RawDialogue(turns=texts, roles=roles, source_id=source_id)


def _open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield from fh
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith((".jsonl", ".json"))]
            if not names:
                raise ValueError(f"No .jsonl/.json file inside {path}")
            for name in names:
                with archive.open(name) as raw:
                    text = raw.read().decode("utf-8", errors="replace")
                yield from text.splitlines()
    else:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            yield from fh


def iter_raw_dialogues(path: str | Path) -> Iterator[RawDialogue]:
    """Stream dialogues from JSONL, JSON array, or ZIP archives.

    Malformed lines are skipped.  A whole-file JSON list is supported too,
    although the primary corpus is JSONL.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    if path.suffix in {".json", ".zip"}:
        # Try whole-file JSON first for .json; fall back to line-delimited.
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                items = payload if isinstance(payload, list) else payload.get("data", [])
                for index, item in enumerate(items):
                    dialogue = _dialogue_from_object(item)
                    if dialogue is not None:
                        dialogue.source_id = dialogue.source_id or str(index)
                        yield dialogue
                return
            except (json.JSONDecodeError, OSError):
                pass
        for line in _open_text(path):
            dialogue = _parse_line(line)
            if dialogue is not None:
                yield dialogue
        return

    for line_number, line in enumerate(_open_text(path), start=1):
        dialogue = _parse_line(line)
        if dialogue is not None:
            if dialogue.source_id is None:
                dialogue.source_id = str(line_number)
            yield dialogue


def _parse_line(line: str) -> RawDialogue | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return _dialogue_from_object(obj)


def download_dataset(config: DataConfig, root: str | Path) -> Path:
    resolved = config.resolve(root)
    return download_file(config.source_url, resolved.raw_path)


def iter_limited(
    dialogues: Iterable[RawDialogue], limit: int | None
) -> Iterator[RawDialogue]:
    if limit is None:
        yield from dialogues
        return
    for index, dialogue in enumerate(dialogues):
        if index >= limit:
            return
        yield dialogue
