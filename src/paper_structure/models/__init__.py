"""
Unified model management for PaperStructure.

All models are hosted on HuggingFace: https://huggingface.co/hpllduck/PaperStructure

Usage:
    from paper_structure.models import registry

    path = registry.get("latex_ocr", "decoder")   # download + resolve
    registry.ensure_all()                          # pre-download everything
    print(registry.status())                       # show what's cached
"""

from .registry import ModelRegistry, registry
from .config import ALL_GROUPS, HF_REPO, LATEX_OCR, YOLOX, PADDLE_OCR

__all__ = [
    "ModelRegistry",
    "registry",
    "ALL_GROUPS",
    "HF_REPO",
    "LATEX_OCR",
    "YOLOX",
    "PADDLE_OCR",
]
