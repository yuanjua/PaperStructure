"""Configuration for LaTeX OCR model."""

from dataclasses import dataclass


@dataclass
class LaTeXOCRConfig:
    """Configuration parameters for LaTeX OCR."""
    max_width: int = 672
    max_height: int = 192
    min_height: int = 32
    min_width: int = 32
    bos_token: int = 1
    max_seq_len: int = 512
    eos_token: int = 2
    temperature: float = 0.00001
