"""Dependency checks with optional automatic installation.

The goal is to make scripts runnable on a fresh Kaggle/Colab kernel without
blindly reinstalling a working CUDA PyTorch build.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Iterable

# import name -> pip requirement
CORE_REQUIREMENTS: dict[str, str] = {
    "torch": "torch",
    "tokenizers": "tokenizers>=0.15,<1",
    "numpy": "numpy>=1.26,<3",
    "yaml": "PyYAML>=6.0,<7",
    "tqdm": "tqdm>=4.66,<5",
    "safetensors": "safetensors>=0.4,<1",
    "requests": "requests>=2.31,<3",
}
DATA_REQUIREMENTS: dict[str, str] = {
    "datasets": "datasets>=2.19,<6",
    "huggingface_hub": "huggingface_hub>=0.23,<1",
}

_pip_checked: set[str] = set()


def missing_packages(requirements: dict[str, str] | None = None) -> list[str]:
    requirements = requirements or {**CORE_REQUIREMENTS, **DATA_REQUIREMENTS}
    return [name for name in requirements if importlib.util.find_spec(name) is None]


def _pip_install(requirement: str) -> None:
    if requirement in _pip_checked:
        return
    print(f"[deps] installing {requirement} ...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", requirement]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not install {requirement!r}.\n"
            f"pip stdout:\n{result.stdout[-2000:]}\npip stderr:\n{result.stderr[-2000:]}"
        )
    _pip_checked.add(requirement)
    print(f"[deps] installed {requirement}", flush=True)


def ensure_dependencies(
    requirements: Iterable[str] | None = None,
    auto_install: bool = True,
) -> list[str]:
    """Return missing import names; install them when auto_install is true."""
    mapping = {**CORE_REQUIREMENTS, **DATA_REQUIREMENTS}
    names = list(requirements) if requirements is not None else list(mapping)
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if not missing:
        return []
    if not auto_install:
        raise ImportError(
            "Missing packages: "
            + ", ".join(missing)
            + f". Install with: {sys.executable} -m pip install -r requirements.txt"
        )
    for name in missing:
        requirement = mapping.get(name, name)
        if name == "torch":
            # Never replace a working CUDA/CPU torch.  This branch only runs when
            # torch itself is absent; the notebook installs the correct wheel.
            requirement = "torch>=2.3,<3"
        _pip_install(requirement)
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"{name} is still not importable after installation")
    return missing


def environment_report() -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["torch_error"] = repr(exc)
    for name in ("tokenizers", "datasets", "huggingface_hub", "numpy", "safetensors"):
        module = importlib.import_module(name) if importlib.util.find_spec(name) else None
        report[name] = getattr(module, "__version__", None) if module else None
    report["missing"] = missing_packages()
    return report
