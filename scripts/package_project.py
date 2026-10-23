"""Create a clean project archive for Kaggle/Colab upload.

Two variants:

* code only (small; the notebook downloads the corpus if Internet is enabled);
* with the raw gzipped corpus (works without Internet, ~272 MB).
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "src",
    "scripts",
    "configs",
    "tests",
    "notebooks",
    "reports",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "build_notebook.py",
    ".gitignore",
]
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "artifacts",
    "exports",
    "dist",
    "data/processed",
}
EXCLUDE_GLOBS = ("*.pyc", "*.part", "*.out", "*.err", "*.zip", "prepare_data.log")


def _skip(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    parts = set(relative.split("/"))
    if parts & EXCLUDE_DIRS:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_GLOBS)


def build(output: Path, with_data: bool) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in INCLUDE:
            source = ROOT / name
            if not source.exists():
                continue
            paths = [source] if source.is_file() else sorted(source.rglob("*"))
            for path in paths:
                if path.is_dir() or _skip(path):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
        if with_data:
            raw = ROOT / "data" / "raw" / "dataset.jsonl.gz"
            if not raw.exists():
                raise FileNotFoundError(f"Raw corpus not found: {raw}")
            # Already gzip-compressed; do not burn time re-compressing 272 MB.
            archive.write(raw, raw.relative_to(ROOT).as_posix(), compress_type=zipfile.ZIP_STORED)
    tmp.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Kaggle/Colab project archives")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--with-data", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    code = build(output_dir / "lyra-chat-project.zip", with_data=False)
    print(f"{code} ({code.stat().st_size / 1e6:.2f} MB)")
    if args.with_data:
        full = build(output_dir / "lyra-chat-with-data.zip", with_data=True)
        print(f"{full} ({full.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
