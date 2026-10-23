"""Environment check; optionally install missing dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import (  # noqa: E402
    DATA_REQUIREMENTS,
    CORE_REQUIREMENTS,
    ensure_dependencies,
    environment_report,
    missing_packages,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Python environment for this project")
    parser.add_argument("--install", action="store_true", help="Install missing packages")
    args = parser.parse_args()
    missing = missing_packages({**CORE_REQUIREMENTS, **DATA_REQUIREMENTS})
    if missing:
        print("Missing packages:", ", ".join(missing))
        if args.install:
            ensure_dependencies(missing, auto_install=True)
        else:
            print("Run with --install to install them automatically.")
    report = environment_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
