# PaperStructure

> Extract structure and content from academic papers with AI-powered layout detection, OCR, and formula recognition.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 🌟 Features

- **📄 Layout Detection** - Detect document elements (titles, sections, paragraphs, formulas, tables, figures)
- **🔤 Text Recognition** - Extract text with PP-OCRv5 (supports 80+ languages)
- **🧮 Formula Recognition** - Convert mathematical formulas to LaTeX
- **📝 Markdown Export** - Generate clean markdown from academic papers
- **⚡ Hardware Acceleration** - GPU/TensorRT support for faster processing
- **🎯 High Accuracy** - State-of-the-art models (YOLOX, DBNet, SVTR)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yuanjua/PaperStructure.git
cd PaperStructure

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Basic Usage

#### Command Line

```bash
# Process a PDF and generate markdown
python -m paper_structure.cli paper.pdf -o output.md

# Process first 5 pages only
python -m paper_structure.cli paper.pdf -o output.md --max-pages 5

# Disable formula recognition for faster processing
python -m paper_structure.cli paper.pdf -o output.md --no-formulas

# Enable verbose output
python -m paper_structure.cli paper.pdf -o output.md -v
```

#### Python API

```python
from paper_structure.pipeline import PaperStructurePipeline

# Initialize pipeline
pipeline = PaperStructurePipeline(
    layout_model="yolox",
    use_formula_recognition=True,
    skip_types=['Page-header', 'Page-footer']
)

# Process PDF
result = pipeline.process_pdf('paper.pdf', page_limit=5)

# Get results
print(f"Detected {result['metadata']['total_elements']} elements")
print(f"Markdown:\n{result['markdown']}")

# Save markdown
pipeline.save_markdown(result, 'output.md')
```

## 📖 Examples

See the [demos/](demos/) folder for detailed examples:

- **[demo_basic.py](demos/demo_basic.py)** - Basic PDF processing
- **[demo_advanced.py](demos/demo_advanced.py)** - Advanced options and customization
- **[demo_images.py](demos/demo_images.py)** - Process images (formulas, tables, documents)
- **[demo_batch.py](demos/demo_batch.py)** - Batch processing multiple PDFs

### Quick Demo

```python
from paper_structure.pipeline import PaperStructurePipeline

pipeline = PaperStructurePipeline()
result = pipeline.process_pdf('paper.pdf', page_limit=2)
pipeline.save_markdown(result, 'output.md')
```

## 🏗️ Architecture

```
PaperStructure Pipeline
├── Layout Detection (YOLOX)
│   ├── Detect document regions
│   ├── Classify element types
│   └── Extract bounding boxes
├── Text Recognition (PP-OCRv5)
│   ├── Detect text regions
│   ├── Classify orientation
│   └── Recognize characters
├── Formula Recognition (LaTeX OCR)
│   ├── Detect formulas
│   └── Convert to LaTeX
└── Markdown Generation
    ├── Structure elements
    └── Format output
```

### Modular Design

All components are modular and can be used independently:

```python
# Use only layout detection
from paper_structure.modules.layout import LayoutDetector
detector = LayoutDetector()
elements = detector.detect(image)

# Use only text recognition
from paper_structure.modules.text import TextRecognizer
recognizer = TextRecognizer()
text_results = recognizer.recognize(image)

# Use only formula recognition
from paper_structure.modules.formula import FormulaRecognizer
formula_rec = FormulaRecognizer()
latex = formula_rec.recognize_region(image, bbox)
```

## 🎯 Supported Element Types

- **Title** - Document title
- **Section-header** - Section headings
- **Text** - Body paragraphs
- **List-item** - Bulleted/numbered lists
- **Formula** - Mathematical equations
- **Table** - Data tables
- **Figure** - Images and charts
- **Caption** - Image/table captions
- **Page-header** - Headers (skippable)
- **Page-footer** - Footers (skippable)

## ⚙️ Configuration

### Pipeline Options

```python
pipeline = PaperStructurePipeline(
    # Layout model selection
    layout_model="yolox",  # or "yolox_tiny", "yolox_quantized"
    
    # Feature toggles
    use_formula_recognition=True,
    
    # Element filtering
    skip_types=['Page-header', 'Page-footer'],
    
    # Hardware acceleration
    use_gpu=False,      # Enable CUDA
    use_dml=False,      # Enable DirectML (Windows)
)
```

### Hardware Acceleration

| Component | TensorRT | CUDA | CPU |
|-----------|----------|------|-----|
| Layout Detection (YOLOX) | ✅ Auto | ✅ Auto | ✅ |
| Text Recognition (OCR) | ✅ | ✅ | ✅ |
| Formula Recognition | ❌ | ✅ | ✅ |

The pipeline automatically selects the best available provider.

## 📊 Performance

| Document Type | Pages/min (CPU) | Pages/min (GPU) |
|--------------|-----------------|-----------------|
| Simple text  | ~2-3 | ~8-10 |
| With formulas | ~1-2 | ~5-7 |
| Complex layout | ~1-2 | ~4-6 |

*Tested on Intel i7-10700K (CPU) and NVIDIA RTX 3080 (GPU)*

## 🔧 Advanced Usage

### Batch Processing

```python
from pathlib import Path
from paper_structure.pipeline import PaperStructurePipeline

pipeline = PaperStructurePipeline()

pdf_files = Path("papers/").glob("*.pdf")
for pdf_file in pdf_files:
    result = pipeline.process_pdf(str(pdf_file))
    output_file = pdf_file.with_suffix('.md')
    pipeline.save_markdown(result, str(output_file))
    print(f"Processed: {pdf_file.name}")
```

### Custom Element Processing

```python
result = pipeline.process_pdf('paper.pdf')

# Access structured data
for page in result['pages']:
    print(f"\nPage {page['page_number']}:")
    for elem in page['elements']:
        print(f"  {elem['type']}: {elem['content'][:50]}...")
        print(f"    Confidence: {elem['confidence']:.2f}")
        print(f"    Bbox: {elem['bbox']}")
```

### GPU Acceleration

```python
# Enable GPU for all components
pipeline = PaperStructurePipeline(
    use_gpu=True,
    use_formula_recognition=True
)

# Text recognition will use CUDA
# Formula recognition will use CUDA
# Layout detection automatically uses TensorRT/CUDA if available
```

## 📦 Package Structure

```
paper_structure/
├── pipeline.py              # Main pipeline
├── cli.py                   # Command-line interface
├── modules/                 # High-level modules
│   ├── layout/             # Layout detection (YOLOX)
│   ├── text/               # Text recognition (PP-OCRv5)
│   ├── formula/            # Formula recognition (LaTeX OCR)
│   └── markdown/           # Markdown generation
└── libs/                    # Low-level libraries
    ├── yolox/              # YOLOX implementation
    ├── onnx_ocr/           # Modular OCR library
    │   ├── text_detector.py
    │   ├── text_classifier.py
    │   ├── text_recognizer.py
    │   └── models/         # Bundled models (21 MB)
    └── latex_ocr/          # LaTeX OCR implementation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project integrates and builds upon several excellent open-source projects:

- **YOLOX** - Layout detection ([Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))
- **PaddleOCR** - Text recognition ([PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR))
- **OnnxOCR** - OCR implementation ([RapidAI/OnnxOCR](https://github.com/RapidAI/OnnxOCR))
- **RapidLaTeXOCR** - Formula recognition ([RapidAI/RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR))
- **unstructured-inference** - Layout analysis ([Unstructured-IO/unstructured-inference](https://github.com/Unstructured-IO/unstructured-inference))

## 📧 Contact

- **Author**: yuanjua
- **Repository**: [github.com/yuanjua/PaperStructure](https://github.com/yuanjua/PaperStructure)

## 🗺️ Roadmap

- [ ] Table structure recognition
- [ ] Multi-column layout support
- [ ] Citation extraction
- [ ] Reference parsing
- [ ] Figure caption matching
- [ ] PDF metadata extraction
- [ ] Export to other formats (HTML, JSON, XML)
- [ ] Web UI
- [ ] REST API
- [ ] Docker container

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐
