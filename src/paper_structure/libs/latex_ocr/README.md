# LaTeX OCR Library

This library contains the LaTeX OCR functionality extracted from [RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR).

## What was extracted

Only the essential components needed for LaTeX formula recognition were extracted:

### Core Modules

- **latex_ocr.py**: Main LaTeX OCR class with auto model downloading
- **models.py**: Encoder-Decoder architecture implementation
- **onnx_session.py**: ONNX Runtime with GPU/TensorRT support (similar to YOLOX)
- **preprocess.py**: Image preprocessing utilities
- **tokenizer.py**: BPE tokenizer for LaTeX formulas
- **config.py**: Configuration dataclass
- **utils.py**: Model downloading utilities

### Features

- ✅ LaTeX formula recognition from images
- ✅ Hardware acceleration support (TensorRT > CUDA > CPU)
- ✅ Automatic model downloading from HuggingFace/GitHub
- ✅ Encoder-decoder architecture with autoregressive generation
- ✅ BPE tokenization for LaTeX output
- ✅ Image preprocessing and resizing

### What was NOT extracted

- Demo scripts and CLI tools
- Testing infrastructure
- Setup.py and old packaging logic
- Documentation files

## Improvements over Original

### 1. GPU/TensorRT Support
```python
# Original: CPU only
session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

# New: Automatic hardware acceleration
session = OrtInferSession(
    model_path,
    use_gpu=True,        # Enable CUDA
    use_tensorrt=True,   # Enable TensorRT
)
# Priority: TensorRT > CUDA > CPU
```

### 2. Simplified Initialization
```python
# Original: Manual path specification
ocr = LaTeXOCR(
    encoder_path="/path/to/encoder.onnx",
    decoder_path="/path/to/decoder.onnx",
    ...
)

# New: Auto-download and simple usage
ocr = LaTeXOCR(use_gpu=True)  # Models downloaded automatically
latex_str, time = ocr(image)
```

### 3. Better Error Handling
- Custom exceptions for ONNX Runtime errors
- Path validation
- Clear error messages

## Usage

```python
from paper_structure.libs.latex_ocr import LaTeXOCR
from PIL import Image

# Initialize (models download automatically on first use)
ocr = LaTeXOCR(
    use_gpu=True,         # Enable CUDA acceleration
    use_tensorrt=False,   # Enable TensorRT (optional)
)

# Recognize formula from image
image = Image.open("formula.png")
latex_string, elapsed_time = ocr(image)

print(f"LaTeX: {latex_string}")
print(f"Time: {elapsed_time:.3f}s")
```

## Models

Models are automatically downloaded from:
- **Repository**: https://github.com/RapidAI/RapidLaTeXOCR
- **Release**: v0.0.0
- **Files**: encoder.onnx, decoder.onnx, image_resizer.onnx, tokenizer.json
- **Location**: `src/paper_structure/libs/latex_ocr/models/`

## Dependencies

- numpy
- opencv-python
- onnxruntime (or onnxruntime-gpu for GPU support)
- Pillow
- tokenizers
- requests
- tqdm

## License

This code is based on RapidLaTeXOCR which is licensed under Apache 2.0.

## Credits

- Original RapidLaTeXOCR: https://github.com/RapidAI/RapidLaTeXOCR
- LaTeX-OCR: https://github.com/lukas-blecher/LaTeX-OCR
