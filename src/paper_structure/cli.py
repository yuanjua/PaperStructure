"""
Command Line Interface for Paper Structure Analysis
"""

import argparse
import sys
from pathlib import Path

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ── Sub-commands ──────────────────────────────────────────────────────────

def _cmd_process(args) -> int:
    """Process a PDF or image and produce markdown / text."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()
    is_image = ext in _IMAGE_EXTS

    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_output.{'txt' if is_image else 'md'}"
    )

    try:
        if is_image:
            return _process_image(input_path, output_path, args)
        elif ext == '.pdf':
            return _process_pdf(input_path, output_path, args)
        else:
            print(
                f"Error: Unsupported file type '{ext}'. "
                f"Supported: .pdf, {', '.join(sorted(_IMAGE_EXTS))}",
                file=sys.stderr,
            )
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _process_pdf(input_path: Path, output_path: Path, args) -> int:
    """Full pipeline for PDFs (layout + OCR + formula)."""
    from .pipeline import PaperStructurePipeline

    if args.verbose:
        print("=" * 70)
        print("Paper Structure Analysis")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Layout Model: {args.layout_model}")
        print(f"Formula Recognition: {'Enabled' if not args.no_formulas else 'Disabled'}")
        print(f"Skip Types: {', '.join(args.skip)}")
        if args.max_pages:
            print(f"Max Pages: {args.max_pages}")
        print("=" * 70)
        print()
        print("Initializing pipeline...")

    pipeline = PaperStructurePipeline(
        layout_model=args.layout_model,
        use_formula_recognition=not args.no_formulas,
        skip_types=args.skip,
    )

    result = pipeline.process_pdf(
        str(input_path),
        page_limit=args.max_pages,
        output_dir=str(output_path.parent),
    )

    pipeline.save_markdown(
        result, str(output_path), save_images=args.save_images
    )

    if args.verbose:
        print(f"Total length: {len(result['markdown'])} characters")
        print("=" * 70)
        print("Processing complete!")
    return 0


def _process_image(input_path: Path, output_path: Path, args) -> int:
    """Direct OCR for images — no layout detection."""
    from .pipeline import OCR

    use_formula = getattr(args, 'formula', False)
    mode = "formula" if use_formula else "text"

    if args.verbose:
        print("=" * 70)
        print(f"OCR ({mode} mode)")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print("=" * 70)
        print()

    ocr = OCR()
    text = ocr(str(input_path), formula=use_formula)

    output_path.write_text(text, encoding='utf-8')

    if args.verbose:
        print(f"Saved to: {output_path}")
        print(f"Length: {len(text)} characters")
        print("=" * 70)
        print("Done!")
    else:
        print(f"Saved to: {output_path}")
    return 0


def _cmd_preview(args) -> int:
    """Process a PDF and generate an annotated preview PDF."""
    from .pipeline import PaperStructurePipeline
    from .preview import generate_preview

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != '.pdf':
        print("Error: preview only supports PDF files", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_preview.pdf")

    if args.verbose:
        print("=" * 70)
        print("Paper Structure Preview")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print("=" * 70)
        print()

    try:
        if args.verbose:
            print("Initializing pipeline...")

        pipeline = PaperStructurePipeline(
            layout_model=args.layout_model,
            use_formula_recognition=not args.no_formulas,
        )

        if args.verbose:
            print("Running detection...\n")

        result = pipeline.process_pdf(
            str(input_path),
            page_limit=args.max_pages,
            output_dir=str(output_path.parent),
        )

        if args.verbose:
            print("\nGenerating preview PDF...")

        generate_preview(result, str(output_path), verbose=args.verbose)

        if args.verbose:
            print("=" * 70)
            print("Preview complete!")
        else:
            print(f"Saved to: {output_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _cmd_models(args) -> int:
    """Manage model weights."""
    from .models import registry

    action = args.action

    if action == "status":
        print(registry.status())
        return 0

    if action == "download":
        if args.group:
            print(f"Downloading model group: {args.group}")
            try:
                registry.ensure_group(args.group)
                print("Done.")
            except KeyError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        else:
            print("Downloading all models...")
            registry.ensure_all()
            print("Done.")
        return 0

    print(f"Unknown action: {action}", file=sys.stderr)
    return 1


# ── Main parser ───────────────────────────────────────────────────────────

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Paper Structure – academic paper analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── process ──
    p_process = subparsers.add_parser(
        "process",
        help="Process a PDF or image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-structure process paper.pdf -o output.md
  paper-structure process photo.png -o output.txt
  paper-structure process formula.png --formula
  paper-structure process paper.pdf --max-pages 3 --save-images
        """,
    )
    p_process.add_argument('input', type=str,
                           help='Input PDF or image file (png, jpg, bmp, tiff, webp)')
    p_process.add_argument('-o', '--output', type=str, default=None,
                           help='Output file path (default: <input>_output.md/.txt)')
    p_process.add_argument('--formula', action='store_true',
                           help='(Images only) Use LaTeX formula recognition instead of text OCR')
    p_process.add_argument('--max-pages', type=int, default=None,
                           help='(PDF only) Maximum number of pages to process')
    p_process.add_argument('--skip', nargs='+', default=['Page-header', 'Page-footer'],
                           help='(PDF only) Element types to skip')
    p_process.add_argument('--no-formulas', action='store_true',
                           help='(PDF only) Disable formula recognition')
    p_process.add_argument('--save-images', action='store_true',
                           help='(PDF only) Save extracted images to an images/ directory')
    p_process.add_argument('--layout-model', type=str, default='yolox',
                           choices=['yolox'],
                           help='(PDF only) Layout detection model (default: yolox)')
    p_process.add_argument('-v', '--verbose', action='store_true',
                           help='Enable verbose output')
    p_process.set_defaults(func=_cmd_process)

    # ── models ──
    p_models = subparsers.add_parser(
        "models",
        help="Manage model weights (download, check status)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-structure models status
  paper-structure models download
  paper-structure models download --group latex_ocr
        """,
    )
    p_models.add_argument('action', choices=['status', 'download'],
                          help='Action to perform')
    p_models.add_argument('--group', type=str, default=None,
                          help='Model group to download (default: all)')
    p_models.set_defaults(func=_cmd_models)

    # ── preview ──
    p_preview = subparsers.add_parser(
        "preview",
        help="Generate an annotated preview PDF with bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-structure preview paper.pdf -o preview.pdf
  paper-structure preview paper.pdf --max-pages 3 -v
        """,
    )
    p_preview.add_argument('input', type=str, help='Input PDF file')
    p_preview.add_argument('-o', '--output', type=str, default=None,
                           help='Output preview PDF path (default: <input>_preview.pdf)')
    p_preview.add_argument('--max-pages', type=int, default=None,
                           help='Maximum number of pages to process')
    p_preview.add_argument('--no-formulas', action='store_true',
                           help='Disable formula recognition')
    p_preview.add_argument('--layout-model', type=str, default='yolox',
                           choices=['yolox'],
                           help='Layout detection model (default: yolox)')
    p_preview.add_argument('-v', '--verbose', action='store_true',
                           help='Enable verbose output')
    p_preview.set_defaults(func=_cmd_preview)

    # ── backwards compat: bare positional arg = process ──
    argv = sys.argv[1:]
    if argv and argv[0] not in ('process', 'models', 'preview', '-h', '--help') and not argv[0].startswith('-'):
        argv_with_cmd = ['process'] + argv
        args = parser.parse_args(argv_with_cmd)
        return args.func(args)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
