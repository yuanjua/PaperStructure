"""
Main Pipeline for Paper Structure Analysis
Orchestrates layout detection, text recognition, and content extraction
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pypdfium2 as pdfium
from PIL import Image
from typing import List, Dict, Any, Optional
from .modules.layout import LayoutDetector
from .modules.text import TextRecognizer
from .modules.formula import FormulaRecognizer
from .modules.markdown import MarkdownGenerator

# Global model management for thread safety
_model_lock = threading.Lock()
_models = {}


class PaperStructurePipeline:
    """
    Complete pipeline for academic paper analysis
    
    Workflow:
    1. Layout Detection (YOLOX) - Detect regions and classify element types
    2. Text Recognition (OnnxOCR) - Extract text from text regions
    3. Formula Recognition (RapidLaTeX) - Extract LaTeX from formula regions
    4. Markdown Generation - Convert to markdown format
    """
    
    def __init__(
        self,
        layout_model: str = "yolox",
        use_formula_recognition: bool = True,
        skip_types: Optional[List[str]] = None,
        use_gpu: bool = False,
        use_dml: bool = False
    ):
        """
        Initialize pipeline
        
        Args:
            layout_model: YOLOX model variant (yolox, yolox_tiny, yolox_quantized)
            use_formula_recognition: Whether to recognize formulas
            skip_types: Element types to skip (e.g., ['Page-header', 'Page-footer'])
            use_gpu: Enable GPU acceleration for OCR (CUDA)
            use_dml: Enable DirectML acceleration for OCR (Windows only)
        
        Hardware Acceleration Summary:
            - Layout (YOLOX): Automatically uses TensorRT > CUDA > CPU
            - Text (OnnxOCR): Controlled by use_gpu and use_dml parameters
            - Formula (RapidLaTeXOCR): Uses CPU with optimized settings
        """
        global _models
        
        # Create unique configuration key for this model set
        config_key = f"{layout_model}_{use_gpu}_{use_dml}_{use_formula_recognition}"
        
        # Thread-safe model initialization with singleton pattern
        with _model_lock:
            if config_key in _models:
                # Reuse existing initialized models
                print("Reusing existing model instances...")
                self.layout_detector = _models[config_key]['layout']
                self.text_recognizer = _models[config_key]['text']
                self.formula_recognizer = _models[config_key]['formula']
            else:
                # Initialize models for the first time
                print("Initializing Paper Structure Pipeline...")
                
                # Initialize modules
                print("  [1/4] Loading layout detector...")
                # YOLOX automatically selects best provider (TensorRT > CUDA > CPU)
                self.layout_detector = LayoutDetector(model_name=layout_model)
                
                print("  [2/4] Loading text recognizer...")
                self.text_recognizer = TextRecognizer(
                    use_angle_cls=False,  # Faster, disable angle classification
                    use_gpu=use_gpu,
                    use_dml=use_dml
                )
                
                print("  [3/4] Loading formula recognizer...")
                if use_formula_recognition:
                    try:
                        self.formula_recognizer = FormulaRecognizer()
                    except ImportError:
                        print("    Warning: Formula recognition not available")
                        self.formula_recognizer = None
                else:
                    self.formula_recognizer = None
                
                # Store models for reuse by other instances
                _models[config_key] = {
                    'layout': self.layout_detector,
                    'text': self.text_recognizer,
                    'formula': self.formula_recognizer
                }
                
                print("  [4/4] Models cached for reuse")
        
        # Initialize markdown generator (lightweight, no caching needed)
        self.markdown_generator = MarkdownGenerator()
        
        # Configuration
        self.skip_types = skip_types or ['Page-header', 'Page-footer']
        
        print("Pipeline ready!\n")
    
    def process_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Process a single image
        
        Args:
            image: PIL Image
            
        Returns:
            List of structured elements with content
        """
        # Step 1: Detect layout
        layout_elements = self.layout_detector.detect(image)
        
        # Step 2: Run OCR once on full image (efficient)
        all_text_results = self.text_recognizer.recognize(image)
        
        # Step 3: Extract content for each element
        structured_elements = []
        
        for element in layout_elements:
            element_type = element['type']
            bbox = element['bbox']
            
            # Skip certain types
            if element_type in self.skip_types:
                continue
            
            # Extract content based on type
            content = ""
            
            if element_type == 'Formula':
                # Use formula recognizer
                if self.formula_recognizer:
                    content = self.formula_recognizer.recognize_region(image, bbox)
                else:
                    content = "[Formula]"
                    
            elif element_type in ['Picture', 'Figure']:
                # Keep as image reference
                content = f"[{element_type}]"
                
            elif element_type == 'Table':
                # Extract text from OCR results
                content = self._extract_text_from_bbox(all_text_results, bbox)
                if not content:
                    content = "[Table]"
                    
            else:
                # Extract text for all other types from OCR results
                content = self._extract_text_from_bbox(all_text_results, bbox)
            
            structured_elements.append({
                'type': element_type,
                'bbox': bbox,
                'content': content,
                'confidence': element['confidence']
            })
        
        return structured_elements
    
    def _extract_text_from_bbox(self, ocr_results: List[Dict[str, Any]], target_bbox: List[int]) -> str:
        """
        Extract text that overlaps with target bbox from OCR results
        
        Args:
            ocr_results: List of OCR detections with bbox and text
            target_bbox: Target bounding box [x1, y1, x2, y2]
            
        Returns:
            Combined text from overlapping regions
        """
        x1, y1, x2, y2 = target_bbox
        target_texts = []
        
        for result in ocr_results:
            rx1, ry1, rx2, ry2 = result['bbox']
            
            # Check if result bbox overlaps with target bbox
            if not (rx2 < x1 or rx1 > x2 or ry2 < y1 or ry1 > y2):
                # Calculate overlap
                overlap_x1 = max(x1, rx1)
                overlap_y1 = max(y1, ry1)
                overlap_x2 = min(x2, rx2)
                overlap_y2 = min(y2, ry2)
                
                overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                result_area = (rx2 - rx1) * (ry2 - ry1)
                
                # If more than 30% overlap, include this text
                if result_area > 0 and overlap_area / result_area > 0.3:
                    target_texts.append((ry1, result['text']))
        
        # Sort by y-coordinate and combine
        target_texts.sort(key=lambda x: x[0])
        return ' '.join(text for _, text in target_texts)
    
    def process_pdf(self, pdf_path: str, page_limit: Optional[int] = None, parallel: bool = True) -> Dict[str, Any]:
        """
        Process entire PDF with optional parallel processing
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum number of pages to process (None = all)
            parallel: Enable parallel processing (one thread per page)
            
        Returns:
            Dictionary with pages and markdown output
        """
        if parallel:
            return self._process_pdf_parallel(pdf_path, page_limit)
        else:
            return self._process_pdf_sequential(pdf_path, page_limit)
    
    def _process_pdf_parallel(self, pdf_path: str, page_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process entire PDF in parallel, with one page per thread
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum number of pages to process (None = all)
            
        Returns:
            Dictionary with pages and markdown output
        """
        print(f"Processing PDF in parallel: {pdf_path}")
        
        # Load PDF and render all pages first
        pdf = pdfium.PdfDocument(pdf_path)
        
        try:
            total_pages = len(pdf)
            pages_to_process = min(page_limit, total_pages) if page_limit is not None else total_pages
            
            print(f"  Total pages: {total_pages}")
            print(f"  Processing: {pages_to_process} pages in parallel\n")
            
            # Pre-allocate results array
            all_pages_results = [None] * pages_to_process
            
            # Render all pages to PIL images first (must be done sequentially)
            print("Rendering pages...")
            images = []
            for i in range(pages_to_process):
                page = pdf[i]
                pil_image = page.render(scale=2.0).to_pil()
                images.append(pil_image)
            pdf.close()
            print(f"  Rendered {len(images)} pages\n")
            
            # Process images concurrently with thread pool
            print("Processing pages in parallel...")
            with ThreadPoolExecutor() as executor:
                # Submit all tasks
                future_to_page = {
                    executor.submit(self._process_page_worker, images[i], i + 1): i 
                    for i in range(pages_to_process)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_page):
                    page_idx = future_to_page[future]
                    try:
                        page_result = future.result()
                        all_pages_results[page_idx] = page_result
                        print(f"  ✓ Completed page {page_result['page_number']}")
                    except Exception as exc:
                        print(f"  ✗ Page {page_idx + 1} generated an exception: {exc}")
            
            # Filter out failed pages and ensure proper ordering
            processed_pages = [p for p in all_pages_results if p is not None]
            processed_pages.sort(key=lambda p: p['page_number'])
            
            print(f"\nSuccessfully processed {len(processed_pages)}/{pages_to_process} pages")
            
            # Generate markdown
            print("Generating markdown...")
            all_elements = []
            for page in processed_pages:
                all_elements.extend(page['elements'])
            
            markdown = self.markdown_generator.generate(all_elements)
            
            return {
                'pages': processed_pages,
                'markdown': markdown,
                'metadata': {
                    'total_pages': total_pages,
                    'processed_pages': len(processed_pages),
                    'total_elements': len(all_elements),
                    'parallel': True
                }
            }
        except Exception as e:
            pdf.close()
            raise e
    
    def _process_page_worker(self, image: Image.Image, page_number: int) -> Dict[str, Any]:
        """
        Worker function for parallel page processing
        
        Args:
            image: Rendered page image
            page_number: Page number (1-indexed)
            
        Returns:
            Dictionary with page number and extracted elements
        """
        # Process the image using the shared models
        elements = self.process_image(image)
        
        return {
            'page_number': page_number,
            'elements': elements,
            'image': image
        }
    
    def _process_pdf_sequential(self, pdf_path: str, page_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process entire PDF sequentially (original implementation)
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum number of pages to process (None = all)
            
        Returns:
            Dictionary with pages and markdown output
        """
        print(f"Processing PDF sequentially: {pdf_path}")
        
        # Load PDF
        pdf = pdfium.PdfDocument(pdf_path)
        
        try:
            total_pages = len(pdf)
            pages_to_process = min(page_limit, total_pages) if page_limit is not None else total_pages
            
            print(f"  Total pages: {total_pages}")
            print(f"  Processing: {pages_to_process} pages\n")
            
            # Process each page
            all_pages = []
            
            for page_num in range(pages_to_process):
                print(f"Page {page_num + 1}/{pages_to_process}...")
                
                # Render page
                page = pdf[page_num]
                pil_image = page.render(scale=2.0).to_pil()
                
                # Process page
                elements = self.process_image(pil_image)
                
                all_pages.append({
                    'page_number': page_num + 1,
                    'elements': elements,
                    'image': pil_image
                })
                
                print(f"  Detected {len(elements)} elements\n")
            
            # Generate markdown
            print("Generating markdown...")
            all_elements = []
            for page in all_pages:
                all_elements.extend(page['elements'])
            
            markdown = self.markdown_generator.generate(all_elements)
            
            return {
                'pages': all_pages,
                'markdown': markdown,
                'metadata': {
                    'total_pages': total_pages,
                    'processed_pages': pages_to_process,
                    'total_elements': len(all_elements),
                    'parallel': False
                }
            }
        finally:
            pdf.close()
    
    def save_markdown(self, result: Dict[str, Any], output_path: str):
        """
        Save markdown to file
        
        Args:
            result: Result from process_pdf
            output_path: Path to save markdown file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['markdown'])
        print(f"Saved markdown to: {output_path}")
    
    def __repr__(self):
        return (
            f"PaperStructurePipeline(\n"
            f"  layout={self.layout_detector},\n"
            f"  text={self.text_recognizer},\n"
            f"  formula={self.formula_recognizer},\n"
            f"  markdown={self.markdown_generator}\n"
            f")"
        )

