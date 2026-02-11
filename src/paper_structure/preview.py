"""
Preview PDF generator: draws detected bounding boxes and semi-transparent
content labels on each page of the original PDF.

Output is a rasterised PDF built entirely with Pillow (no extra dependencies).
"""

from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Colour palette per element type  (R, G, B)
# ---------------------------------------------------------------------------
_TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Title":          (41,  98, 255),   # blue
    "Section-header": (41,  98, 255),
    "Text":           (56, 142,  60),   # green
    "List-item":      (56, 142,  60),
    "Formula":        (211, 47,  47),   # red
    "Table":          (239, 108,  0),   # orange
    "Picture":        (142,  36, 170),  # purple
    "Figure":         (142,  36, 170),
    "Caption":        (0,  137, 123),   # teal
    "Footnote":       (120, 120, 120),  # gray
    "Page-header":    (180, 180, 180),
    "Page-footer":    (180, 180, 180),
}
_DEFAULT_COLOR = (100, 100, 100)

# Fill opacity (0–255)
_FILL_ALPHA = 40
# Border width in pixels
_BORDER_WIDTH = 3
# Max chars for content preview inside the bbox
_MAX_CONTENT_CHARS = 80


def _get_color(element_type: str) -> Tuple[int, int, int]:
    return _TYPE_COLORS.get(element_type, _DEFAULT_COLOR)


def _truncate(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Truncate text and add ellipsis if too long."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a reasonable font; fall back to default bitmap font."""
    # Common paths for a readable monospace / sans font
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_page(
    page_image: Image.Image,
    elements: List[Dict[str, Any]],
) -> Image.Image:
    """Draw bounding boxes and content labels on a single page image.

    Returns a new RGB image suitable for PDF export.
    """
    base = page_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    label_font = _load_font(20)
    content_font = _load_font(16)

    for elem in elements:
        bbox = elem["bbox"]
        x1, y1, x2, y2 = bbox
        etype = elem.get("type", "Unknown")
        confidence = elem.get("confidence", 0.0)
        content = elem.get("content", "")
        color = _get_color(etype)

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, _FILL_ALPHA))

        # Solid border
        for i in range(_BORDER_WIDTH):
            draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=(*color, 200))

        # Label: "Type (confidence%)" at top-left
        label = f"{etype} ({confidence:.0%})"
        lbox = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = lbox[2] - lbox[0], lbox[3] - lbox[1]
        # Label background
        draw.rectangle(
            [x1, y1, x1 + lw + 8, y1 + lh + 6],
            fill=(*color, 180),
        )
        draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255, 240), font=label_font)

        # Content preview inside bbox (below the label)
        if content and not content.startswith("__IMAGE__"):
            preview = _truncate(content)
            if etype == "Formula":
                preview = f"$ {preview}"

            text_y = y1 + lh + 10
            box_w = x2 - x1 - 8
            box_h = y2 - (text_y + 4)

            if box_w > 30 and box_h > 14:
                # Word-wrap the preview text to fit inside the bbox
                lines = _wrap_text(preview, content_font, box_w, draw)
                for line in lines:
                    if text_y + 18 > y2 - 4:
                        break
                    # draw.text(
                        (x1 + 4, text_y),
                        line,
                        fill=(80, 80, 80, 180),
                        font=content_font,
                    # )
                    text_y += 18

    # Composite and convert back to RGB for PDF saving
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> List[str]:
    """Simple word-wrap that respects pixel width."""
    words = text.split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [text[:20]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_preview(
    result: Dict[str, Any],
    output_path: str,
    verbose: bool = False,
) -> None:
    """Generate an annotated preview PDF from pipeline results.

    Args:
        result: The dict returned by ``PaperStructurePipeline.process_pdf``.
                Must contain ``pages`` with ``image`` and ``elements`` keys.
        output_path: Where to save the preview PDF.
        verbose: Print progress info.
    """
    pages = result.get("pages", [])
    if not pages:
        raise ValueError("No pages in result to preview.")

    annotated: List[Image.Image] = []

    for page in pages:
        page_num = page["page_number"]
        image: Image.Image = page["image"]
        elements: List[Dict[str, Any]] = page["elements"]

        if verbose:
            print(f"  Annotating page {page_num} ({len(elements)} elements)...")

        annotated_page = _draw_page(image, elements)
        annotated.append(annotated_page)

    # Save as multi-page PDF
    if verbose:
        print(f"  Saving preview PDF ({len(annotated)} pages)...")

    annotated[0].save(
        output_path,
        format="PDF",
        save_all=True,
        append_images=annotated[1:],
        resolution=150.0,
    )

    if verbose:
        print(f"  Preview saved to: {output_path}")
