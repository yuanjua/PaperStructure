"""
Modular OCR Library extracted from OnnxOCR

Three independent stages:
- TextDetector: Finds text regions in images
- TextClassifier: Corrects text orientation
- TextRecognizer: Converts text images to strings

Each module supports batch processing and has its own pre/post-processing.

High-level interface:
- OCRPipeline: Complete OCR pipeline (detection + classification + recognition)
- ONNXPaddleOcr: Alias for OCRPipeline (for compatibility)
"""

from .text_detector import TextDetector
from .text_classifier import TextClassifier
from .text_recognizer import TextRecognizer
from .config import DetectorConfig, ClassifierConfig, RecognizerConfig
from .pipeline import OCRPipeline, ONNXPaddleOcr

__all__ = [
    "TextDetector",
    "TextClassifier", 
    "TextRecognizer",
    "DetectorConfig",
    "ClassifierConfig",
    "RecognizerConfig",
    "OCRPipeline",
    "ONNXPaddleOcr",
]
