# YOLOX Layout Detection Library

This library contains the YOLOX layout detection functionality extracted from [unstructured-inference](https://github.com/Unstructured-IO/unstructured-inference).

## What was extracted

Only the essential components needed for YOLOX-based layout detection were extracted:

### Core Modules

- **model.py**: Model factory and singleton pattern for managing model instances
- **yolox_model.py**: YOLOX model implementation with ONNX runtime
- **layoutelement.py**: Data structures for layout elements and their collections
- **elements.py**: Basic geometric elements (Rectangle, TextRegion, etc.)
- **constants.py**: Element types and constants
- **utils.py**: Utility functions for lazy evaluation and model loading
- **math_utils.py**: Safe mathematical operations

### Features

- ✅ YOLOX ONNX model inference
- ✅ Hardware acceleration support (TensorRT > CUDA > CPU)
- ✅ Multiple model variants (yolox, yolox_tiny, yolox_quantized)
- ✅ Automatic model downloading from HuggingFace Hub
- ✅ Layout element detection with bounding boxes
- ✅ Element classification (Caption, Footnote, Formula, List-item, etc.)

### What was NOT extracted

- Detectron2 models
- Table extraction/parsing
- OCR functionality
- Image preprocessing utilities beyond what's needed for YOLOX
- Training code
- Test utilities

## Usage

```python
from paper_structure.libs.yolox import get_model
from PIL import Image

# Load model (automatically downloads on first use)
model = get_model("yolox")  # or "yolox_tiny" or "yolox_quantized"

# Detect layout elements
image = Image.open("document.jpg")
layout_elements = model.predict(image)

# Access detected elements
for element in layout_elements.iter_elements():
    print(f"Type: {element.type}")
    print(f"Bbox: {element.bbox.coordinates}")
    print(f"Confidence: {element.prob}")
```

## Dependencies

- numpy>=1.20.0
- opencv-python>=4.5.0
- onnxruntime>=1.10.0
- Pillow>=8.0.0
- huggingface-hub>=0.10.0

## License

This code is based on unstructured-inference which is licensed under Apache 2.0.
The original YOLOX code is from Megvii, Inc. and its affiliates.

## Credits

- Original unstructured-inference: https://github.com/Unstructured-IO/unstructured-inference
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
