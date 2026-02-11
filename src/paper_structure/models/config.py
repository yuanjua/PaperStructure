"""
Model definitions: sources, filenames, and versions.

Single source of truth for every model weight used in the project.
All models are hosted on a single HuggingFace repository.
"""

from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# The one repo that holds everything
# ---------------------------------------------------------------------------
HF_REPO = "hpllduck/PaperStructure"


@dataclass(frozen=True)
class ModelFile:
    """A single model file inside the HuggingFace repo."""
    filename: str          # path inside the repo, e.g. "latex_ocr/decoder.onnx"
    description: str = ""


@dataclass(frozen=True)
class ModelGroup:
    """A logical group of model files that belong together."""
    name: str
    description: str
    files: Dict[str, ModelFile]  # key -> ModelFile

    @property
    def filenames(self) -> List[str]:
        return [f.filename for f in self.files.values()]


# ---------------------------------------------------------------------------
# LaTeX OCR — formula recognition
# ---------------------------------------------------------------------------
LATEX_OCR = ModelGroup(
    name="latex_ocr",
    description="RapidLaTeXOCR encoder-decoder models for formula recognition",
    files={
        "decoder": ModelFile(
            filename="latex_ocr/decoder.onnx",
            description="Transformer decoder",
        ),
        "encoder": ModelFile(
            filename="latex_ocr/encoder.onnx",
            description="ViT encoder",
        ),
        "image_resizer": ModelFile(
            filename="latex_ocr/image_resizer.onnx",
            description="Adaptive image resizer",
        ),
        "tokenizer": ModelFile(
            filename="latex_ocr/tokenizer.json",
            description="BPE tokenizer vocabulary",
        ),
    },
)

# ---------------------------------------------------------------------------
# YOLOX — document layout detection
# ---------------------------------------------------------------------------
YOLOX = ModelGroup(
    name="yolox",
    description="YOLOX layout detection models (from unstructured-inference)",
    files={
        "yolox": ModelFile(
            filename="yolox/yolox_l0.05.onnx",
            description="YOLOX-L default model",
        ),
    },
)

# ---------------------------------------------------------------------------
# PaddleOCR — text detection / classification / recognition
# ---------------------------------------------------------------------------
PADDLE_OCR = ModelGroup(
    name="paddle_ocr",
    description="PP-OCRv5 text detection / classification / recognition",
    files={
        "detector": ModelFile(
            filename="paddle_ocr/det.onnx",
            description="DB text detector",
        ),
        "classifier": ModelFile(
            filename="paddle_ocr/cls.onnx",
            description="Text angle classifier",
        ),
        "recognizer": ModelFile(
            filename="paddle_ocr/rec.onnx",
            description="SVTR text recognizer",
        ),
        "dictionary": ModelFile(
            filename="paddle_ocr/ppocrv5_dict.txt",
            description="Character dictionary (6k+ chars)",
        ),
    },
)

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
ALL_GROUPS: Dict[str, ModelGroup] = {
    "latex_ocr": LATEX_OCR,
    "yolox": YOLOX,
    "paddle_ocr": PADDLE_OCR,
}
