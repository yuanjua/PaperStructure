"""
Main Pipeline for Paper Structure Analysis
Orchestrates layout detection, text recognition, and content extraction
"""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
        
        # Image output directory (set during process_pdf)
        self._image_output_dir: Optional[Path] = None
        self._extracted_images: List[Dict[str, Any]] = []
        
        print("Pipeline ready!\n")
    
    def process_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Process a single image with batched formula recognition
        
        Args:
            image: PIL Image
            
        Returns:
            List of structured elements with content
        """
        # Step 1: Detect layout
        layout_elements = self.layout_detector.detect(image)
        
        # Step 2: Run OCR once on full image (efficient)
        all_text_results = self.text_recognizer.recognize(image)
        
        # Step 3: Sort elements by reading order (top-to-bottom, left-to-right)
        # For two-column layouts, we need to detect columns and sort within each
        layout_elements = self._sort_by_reading_order(layout_elements, image.width)
        
        # Step 4: Collect formula regions for batch processing
        formula_indices = []  # Track which elements are formulas
        formula_images = []   # Cropped formula images
        
        # Step 5: Extract content for each element (defer formula recognition)
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
                # Collect formula for batch processing
                if self.formula_recognizer:
                    formula_image = self.formula_recognizer.crop_region(image, bbox)
                    formula_indices.append(len(structured_elements))
                    formula_images.append(formula_image)
                    content = "__FORMULA_PLACEHOLDER__"  # Will be replaced after batch
                else:
                    content = "[Formula]"
                    
            elif element_type in ['Picture', 'Figure']:
                # Extract and save image, store path for later
                content = self._extract_image_region(image, bbox)
                
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
        
        # Step 6: Batch process all formulas at once
        if formula_images and self.formula_recognizer:
            formula_results = self.formula_recognizer.recognize_batch(formula_images)
            for idx, latex_str in zip(formula_indices, formula_results):
                structured_elements[idx]['content'] = latex_str
        
        return structured_elements
    
    def _extract_image_region(self, image: Image.Image, bbox: List[int]) -> str:
        """
        Extract image region and return a placeholder with hash for later saving
        
        Args:
            image: Full PIL Image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Image placeholder string with hash identifier
        """
        x1, y1, x2, y2 = bbox
        
        # Add small padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Crop the region
        region = image.crop((x1, y1, x2, y2))
        
        # Generate hash from image content for unique filename
        import io
        img_bytes = io.BytesIO()
        region.save(img_bytes, format='JPEG', quality=95)
        img_hash = hashlib.sha256(img_bytes.getvalue()).hexdigest()
        
        # Store image data for later saving
        self._extracted_images.append({
            'hash': img_hash,
            'image': region,
            'bbox': bbox
        })
        
        # Return placeholder that will be used in markdown
        return f"__IMAGE__{img_hash}__"
    
    def _save_extracted_images(self, output_dir: Path) -> Dict[str, str]:
        """
        Save all extracted images to output directory
        
        Args:
            output_dir: Directory to save images
            
        Returns:
            Mapping of image hash to relative file path
        """
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        hash_to_path = {}
        for img_data in self._extracted_images:
            img_hash = img_data['hash']
            if img_hash not in hash_to_path:
                filename = f"{img_hash}.jpg"
                filepath = images_dir / filename
                img_data['image'].save(filepath, 'JPEG', quality=95)
                hash_to_path[img_hash] = f"images/{filename}"
        
        return hash_to_path
    
    def _sort_by_reading_order(self, elements: List[Dict[str, Any]], page_width: int) -> List[Dict[str, Any]]:
        """
        Sort elements by reading order, handling multi-column layouts
        
        Args:
            elements: List of layout elements
            page_width: Width of the page in pixels
            
        Returns:
            Elements sorted in reading order
        """
        if not elements:
            return elements
        
        # Detect if this is a two-column layout by analyzing element positions
        # Elements in left column have x_center < page_width/2
        # Elements in right column have x_center >= page_width/2
        
        mid_x = page_width / 2
        column_threshold = page_width * 0.1  # 10% tolerance for column detection
        
        left_column = []
        right_column = []
        full_width = []  # Elements spanning both columns (titles, headers, etc.)
        
        for element in elements:
            bbox = element['bbox']
            x1, y1, x2, y2 = bbox
            elem_width = x2 - x1
            x_center = (x1 + x2) / 2
            
            # Check if element spans most of the page width (full-width element)
            if elem_width > page_width * 0.6:
                full_width.append(element)
            # Check if element is in left column
            elif x2 < mid_x + column_threshold:
                left_column.append(element)
            # Check if element is in right column
            elif x1 > mid_x - column_threshold:
                right_column.append(element)
            else:
                # Element spans columns but isn't full-width, treat as full-width
                full_width.append(element)
        
        # Sort each group by y-coordinate (top to bottom)
        def sort_key(elem):
            return elem['bbox'][1]  # y1 coordinate
        
        left_column.sort(key=sort_key)
        right_column.sort(key=sort_key)
        full_width.sort(key=sort_key)
        
        # Merge: full-width elements first (by position), then left column, then right column
        # But we need to interleave based on y-position
        result = []
        
        # Create a list of (y_position, element, priority) tuples
        # Priority: 0 = full-width (highest), 1 = left column, 2 = right column
        all_elements = []
        for elem in full_width:
            all_elements.append((elem['bbox'][1], elem, 0))
        for elem in left_column:
            all_elements.append((elem['bbox'][1], elem, 1))
        for elem in right_column:
            all_elements.append((elem['bbox'][1], elem, 2))
        
        # Sort by y-position, then by priority (full-width first, then left, then right)
        # For two-column layout, we want: full-width at top, then all left column, then all right column
        # But if columns are interleaved with full-width, respect that
        
        # Group elements by vertical regions
        # A full-width element creates a "break" between column sections
        
        sorted_result = []
        current_left = []
        current_right = []
        
        # Sort all elements by y-position
        all_sorted = sorted(all_elements, key=lambda x: x[0])
        
        for y_pos, elem, priority in all_sorted:
            if priority == 0:  # Full-width element
                # Flush current columns (left first, then right)
                sorted_result.extend(current_left)
                sorted_result.extend(current_right)
                current_left = []
                current_right = []
                # Add full-width element
                sorted_result.append(elem)
            elif priority == 1:  # Left column
                current_left.append(elem)
            else:  # Right column
                current_right.append(elem)
        
        # Flush remaining column elements
        sorted_result.extend(current_left)
        sorted_result.extend(current_right)
        
        return sorted_result
    
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
    
    def process_pdf(self, pdf_path: str, page_limit: Optional[int] = None, parallel: bool = True, output_dir: Optional[str] = None, max_workers: int = 8) -> Dict[str, Any]:
        """
        Process entire PDF with optional parallel processing
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum number of pages to process (None = all)
            parallel: Enable parallel processing (one thread per page)
            output_dir: Directory to save extracted images (default: same as PDF)
            max_workers: Maximum number of parallel workers (default: 8)
            
        Returns:
            Dictionary with pages and markdown output
        """
        # Reset extracted images for new processing
        self._extracted_images = []
        
        # Set output directory for images
        pdf_path_obj = Path(pdf_path)
        if output_dir:
            self._image_output_dir = Path(output_dir)
        else:
            self._image_output_dir = pdf_path_obj.parent
        
        if parallel:
            return self._process_pdf_parallel(pdf_path, page_limit, max_workers=max_workers)
        else:
            return self._process_pdf_sequential(pdf_path, page_limit)
    
    def _process_pdf_parallel(self, pdf_path: str, page_limit: Optional[int] = None, max_workers: int = 8) -> Dict[str, Any]:
        """
        Process entire PDF in parallel with memory-optimized on-the-fly rendering
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum number of pages to process (None = all)
            max_workers: Maximum number of parallel workers (default: 8)
            
        Returns:
            Dictionary with pages and markdown output
        """
        print(f"Processing PDF in parallel: {pdf_path}")
        
        # Load PDF
        pdf = pdfium.PdfDocument(pdf_path)
        
        try:
            total_pages = len(pdf)
            pages_to_process = min(page_limit, total_pages) if page_limit is not None else total_pages
            
            # Adjust max_workers based on page count
            effective_workers = min(max_workers, pages_to_process)
            
            print(f"  Total pages: {total_pages}")
            print(f"  Processing: {pages_to_process} pages (max {effective_workers} workers)\n")
            
            # Pre-allocate results array
            all_pages_results = [None] * pages_to_process
            
            # Memory-optimized: render and process pages on-the-fly
            # Use a lock for thread-safe PDF page rendering
            pdf_lock = threading.Lock()
            
            def render_and_process_page(page_idx: int) -> Dict[str, Any]:
                """Render a single page and process it (thread-safe)"""
                # Thread-safe page rendering
                with pdf_lock:
                    page = pdf[page_idx]
                    pil_image = page.render(scale=2.0).to_pil()
                
                # Process the image (can run in parallel)
                elements = self.process_image(pil_image)
                
                return {
                    'page_number': page_idx + 1,
                    'elements': elements,
                    'image': pil_image  # Keep for potential image extraction
                }
            
            # Process pages concurrently with bounded thread pool
            print("Processing pages in parallel (on-the-fly rendering)...")
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Submit all tasks
                future_to_page = {
                    executor.submit(render_and_process_page, i): i 
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
            
            # Close PDF after all pages are processed
            pdf.close()
            
            # Filter out failed pages and ensure proper ordering
            processed_pages = [p for p in all_pages_results if p is not None]
            processed_pages.sort(key=lambda p: p['page_number'])
            
            print(f"\nSuccessfully processed {len(processed_pages)}/{pages_to_process} pages")
            
            # Save extracted images and get path mapping
            print("Saving extracted images...")
            hash_to_path = self._save_extracted_images(self._image_output_dir)
            
            # Generate markdown
            print("Generating markdown...")
            all_elements = []
            for page in processed_pages:
                all_elements.extend(page['elements'])
            
            markdown = self.markdown_generator.generate(all_elements)
            
            # Replace image placeholders with actual paths
            for img_hash, img_path in hash_to_path.items():
                placeholder = f"__IMAGE__{img_hash}__"
                markdown = markdown.replace(f"![Figure]\n\n{placeholder}", f"![](images/{img_hash}.jpg)")
                markdown = markdown.replace(placeholder, f"![](images/{img_hash}.jpg)")
            
            return {
                'pages': processed_pages,
                'markdown': markdown,
                'metadata': {
                    'total_pages': total_pages,
                    'processed_pages': len(processed_pages),
                    'total_elements': len(all_elements),
                    'extracted_images': len(hash_to_path),
                    'parallel': True,
                    'max_workers': effective_workers
                }
            }
        except Exception as e:
            pdf.close()
            raise e
    
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
            
            # Save extracted images and get path mapping
            print("Saving extracted images...")
            hash_to_path = self._save_extracted_images(self._image_output_dir)
            
            # Generate markdown
            print("Generating markdown...")
            all_elements = []
            for page in all_pages:
                all_elements.extend(page['elements'])
            
            markdown = self.markdown_generator.generate(all_elements)
            
            # Replace image placeholders with actual paths
            for img_hash, img_path in hash_to_path.items():
                placeholder = f"__IMAGE__{img_hash}__"
                markdown = markdown.replace(f"![Figure]\n\n{placeholder}", f"![](images/{img_hash}.jpg)")
                markdown = markdown.replace(placeholder, f"![](images/{img_hash}.jpg)")
            
            return {
                'pages': all_pages,
                'markdown': markdown,
                'metadata': {
                    'total_pages': total_pages,
                    'processed_pages': pages_to_process,
                    'total_elements': len(all_elements),
                    'extracted_images': len(hash_to_path),
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

