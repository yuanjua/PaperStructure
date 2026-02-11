"""
Paper Structure Analysis Library
Modular pipeline for academic paper parsing with ONNX models
"""

from .pipeline import PaperStructurePipeline, OCR
from .modules.layout import LayoutDetector
from .modules.text import TextRecognizer
from .modules.formula import FormulaRecognizer
from .modules.markdown import MarkdownGenerator

__version__ = "0.1.0"
__all__ = [
    'PaperStructurePipeline',
    'OCR',
    'LayoutDetector',
    'TextRecognizer',
    'FormulaRecognizer',
    'MarkdownGenerator',
]
