"""ClinicalShift-2026: Multi-agent LLM robustness under clinical dataset shift."""

__version__ = "0.1.0"


def get_device() -> str:
    """Detect best available device for PyTorch inference.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
