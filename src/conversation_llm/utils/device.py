"""Device and mixed-precision selection with explicit, readable errors."""

from __future__ import annotations

import torch


def resolve_device_name(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    if requested.startswith("mps") and not getattr(torch.backends, "mps", None):
        raise RuntimeError("MPS was requested but is not available in this PyTorch build.")
    return requested


def resolve_device(requested: str = "auto") -> torch.device:
    name = resolve_device_name(requested)
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device specification: {requested!r}") from exc
    if device.type == "cuda":
        # Fail early with an actionable message instead of a kernel error later.
        torch.cuda.init()
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
    return device


def describe_device(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    else:
        info["cpu_count"] = torch.get_num_threads()
    return info


def amp_settings(
    device: torch.device, requested: str = "auto"
) -> tuple[bool, torch.dtype | None, bool]:
    """Return (enabled, autocast dtype, grad_scaler_needed).

    BF16 has the same exponent range as FP32 and normally needs no scaler.  FP16
    needs a GradScaler.  On CPU we keep FP32 by default; BF16 CPU autocast is
    opt-in via `amp: bf16` because it depends on the CPU model.
    """
    requested = (requested or "auto").lower()
    if requested == "off":
        return False, None, False

    if device.type == "cuda":
        if requested == "fp16":
            return True, torch.float16, True
        if requested == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 was requested but this GPU does not support it")
            return True, torch.bfloat16, False
        # auto: prefer BF16 on Ampere+, FP16 elsewhere
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True

    if requested == "fp16":
        raise RuntimeError("FP16 autocast on CPU is not supported; use 'off' or 'bf16'")
    if requested == "bf16":
        return True, torch.bfloat16, False
    return False, None, False
