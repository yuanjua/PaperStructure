# PaperStructure

PaperStructure is a lightweight CLI tool designed to transform academic papers into clean, structured Markdown. By leveraging ONNX models, it delivers high-performance inference optimized for standard laptops. It is a reliable companion for formula-heavy research, though users may currently observe lower accuracy in vertical table recognition.

## Features

- **Layout Detection** -- YOLOX detects titles, sections, paragraphs, formulas, tables, figures
- **Text Recognition** -- PP-OCRv5 ONNX pipeline
- **Formula Recognition** -- Encoder-decoder LaTeX OCR
- **Markdown Export** -- clean, readable markdown output
- **Parallel Processing** -- multi-threaded PDF page processing

## Demo

| PDF | Markdown |  
|-----------|-----------|  
| <img width="597" height="813" alt="Screenshot 2026-02-11 at 21 54 00" src="https://github.com/user-attachments/assets/63532903-dbdd-4ae9-afd2-95d41833d4fa" /> | <img width="691" height="795" alt="Screenshot 2026-02-11 at 22 20 52" src="https://github.com/user-attachments/assets/15e955ab-9826-4f24-9a7a-64abf203e15d" /> |  
 
## Installation

```bash
pip install -e .
```

This registers the `paper-structure` CLI and installs the Python package.

## CLI Usage

```bash
# Process a PDF
paper-structure process paper.pdf -o output.md

# Shorthand also works:
paper-structure paper.pdf -o output.md

# Process first 5 pages, verbose, no images to save to disk
paper-structure process paper.pdf -o output.md --max-pages 5 -v --no-images

# Generate annotated preview PDF with bounding boxes
paper-structure preview paper.pdf -o preview.pdf

# Manage models
paper-structure models status      # show what's downloaded
paper-structure models download    # pre-download all models
```

## Python API

```python
from paper_structure import PaperStructurePipeline

pipeline = PaperStructurePipeline()
result = pipeline.process_pdf("paper.pdf")

# Markdown string
print(result["markdown"])

# Save to file
pipeline.save_markdown(result, "output.md")

# Structured page data
for page in result["pages"]:
    for elem in page["elements"]:
        print(elem["type"], elem["content"][:80])
```

### Options

```python
pipeline = PaperStructurePipeline(
    layout_model="yolox",              # layout detection model
    use_formula_recognition=True,      # enable LaTeX formula OCR
    skip_types=["Page-header", "Page-footer"],
    use_gpu=False,                     # CUDA acceleration for OCR
    use_dml=False,                     # DirectML (Windows)
)

result = pipeline.process_pdf(
    "paper.pdf",
    page_limit=5,                      # max pages to process
    max_workers=8,                     # parallel threads
)
```

### Model Management

```python
from paper_structure.models import registry

registry.ensure_all()       # pre-download everything
print(registry.status())    # show cache status
```

## Models

The tool automatically downloads models on its first call. All model weights are hosted at [`hpllduck/PaperStructure`](https://huggingface.co/hpllduck/PaperStructure) (~399 MB total) and cached locally via `huggingface_hub`.

| Group | Files | Description |
|-------|-------|-------------|
| `latex_ocr` | encoder, decoder, image_resizer, tokenizer | RapidLaTeXOCR formula recognition |
| `yolox` | yolox_l0.05.onnx | YOLOX-L document layout detection |
| `paddle_ocr` | det, cls, rec, dictionary | PP-OCRv5 text detection/recognition |

## License

Apache License 2.0. Individual model weights retain their original licenses (MIT for LaTeX OCR, Apache-2.0 for YOLOX and PaddleOCR).

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) -- text recognition
- [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) -- ONNX OCR pipeline
- [RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR) -- formula recognition
- [unstructured-inference](https://github.com/Unstructured-IO/unstructured-inference) -- layout detection
