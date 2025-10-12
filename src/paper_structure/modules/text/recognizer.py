"""
Text Recognition Module
Uses internal modular OCR library for text extraction from detected regions
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any

from paper_structure.libs.onnx_ocr import ONNXPaddleOcr


class TextRecognizer:
    """
    Text recognition using modular OCR (PP-OCRv5)
    
    Hardware acceleration options:
    - use_gpu: Enable CUDA acceleration (default: False)
    - use_dml: Enable DirectML acceleration on Windows (default: False)
    """
    
    def __init__(self, use_angle_cls: bool = False, use_gpu: bool = False, use_dml: bool = False):
        """
        Initialize text recognizer
        
        Args:
            use_angle_cls: Whether to use angle classification
            use_gpu: Enable CUDA GPU acceleration
            use_dml: Enable DirectML acceleration (Windows only)
        """
        # Initialize with explicit parameters using internal libs
        self.ocr = ONNXPaddleOcr(
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            use_dml=use_dml
        )
        
    def recognize(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Recognize text in image
        
        Args:
            image: PIL Image
            
        Returns:
            List of text regions with bbox, text, and confidence
        """
        # Convert to numpy array for OnnxOCR (BGR format)
        img = np.array(image)
        img = img[:, :, ::-1].copy()  # RGB to BGR
        
        # Run OCR
        result = self.ocr.ocr(img)
        
        # Extract results
        boxes = result[0] if result and len(result) > 0 else []
        
        # Convert to standard format
        results = []
        for detection in boxes:
            box, (text, confidence) = detection
            # Convert box coordinates
            box_array = np.array(box).astype(int)
            x1 = int(min(box_array[:, 0]))
            y1 = int(min(box_array[:, 1]))
            x2 = int(max(box_array[:, 0]))
            y2 = int(max(box_array[:, 1]))
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'text': text,
                'confidence': float(confidence)
            })
        
        return results
    
    def recognize_region(self, image: Image.Image, bbox: List[int]) -> str:
        """
        Recognize text in specific region
        
        Args:
            image: Full PIL Image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Recognized text
        """
        # Get all text from full image first (more reliable)
        all_results = self.recognize(image)
        
        # Filter results that overlap with target bbox
        x1, y1, x2, y2 = bbox
        target_texts = []
        
        for result in all_results:
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
                    target_texts.append((ry1, result['text']))  # Store with y-coord for sorting
        
        # Sort by y-coordinate and combine
        target_texts.sort(key=lambda x: x[0])
        return ' '.join(text for _, text in target_texts)
    
    def __repr__(self):
        return "TextRecognizer(model=PP-OCRv5)"
