"""
Command Line Interface for Paper Structure Analysis
"""

import argparse
import sys
from pathlib import Path
from .pipeline import PaperStructurePipeline


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract structure and content from academic papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF and save to markdown
  python -m paper_structure.cli input.pdf -o output.md
  
  # Process first 3 pages only
  python -m paper_structure.cli input.pdf -o output.md --max-pages 3
  
  # Skip headers and footers
  python -m paper_structure.cli input.pdf -o output.md --skip Page-header Page-footer
  
  # Disable formula recognition for faster processing
  python -m paper_structure.cli input.pdf -o output.md --no-formulas
        """
    )
    
    # Input/Output
    parser.add_argument(
        'input',
        type=str,
        help='Input PDF file path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output markdown file path (default: <input>_output.md)'
    )
    
    # Processing options
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to process (default: all pages)'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        default=['Page-header', 'Page-footer'],
        help='Element types to skip (default: Page-header Page-footer)'
    )
    parser.add_argument(
        '--no-formulas',
        action='store_true',
        help='Disable formula recognition (faster but less accurate)'
    )
    
    # Model options
    parser.add_argument(
        '--layout-model',
        type=str,
        default='yolox',
        choices=['yolox', 'yolox_tiny', 'yolox_quantized'],
        help='Layout detection model (default: yolox)'
    )
    
    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    if not input_path.suffix.lower() == '.pdf':
        print(f"Error: Input file must be a PDF", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_output.md")
    else:
        output_path = Path(args.output)
    
    # Print configuration
    if args.verbose:
        print("=" * 70)
        print("Paper Structure Analysis CLI")
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
    
    try:
        # Initialize pipeline
        if args.verbose:
            print("Initializing pipeline...")
        
        pipeline = PaperStructurePipeline(
            layout_model=args.layout_model,
            use_formula_recognition=not args.no_formulas,
            skip_types=args.skip
        )
        
        if args.verbose:
            print("Pipeline ready!")
            print()
        
        # Process PDF
        if args.verbose:
            print(f"Processing PDF: {input_path.name}")
        
        result = pipeline.process_pdf(
            str(input_path),
            page_limit=args.max_pages
        )
        markdown = result['markdown']
        
        # Save output
        output_path.write_text(markdown, encoding='utf-8')
        
        if args.verbose:
            print(f"\nMarkdown saved to: {output_path}")
            print(f"Total length: {len(markdown)} characters")
            print("=" * 70)
            print("Processing complete!")
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


if __name__ == '__main__':
    sys.exit(main())

