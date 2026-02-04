"""
Formula Recognition Module
Using extracted LaTeX OCR library with GPU/TensorRT support
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

from paper_structure.libs.latex_ocr import LaTeXOCR
from paper_structure.libs.latex_ocr.config import LaTeXOCRConfig


class FormulaRecognizer:
    """
    LaTeX Formula Recognition with GPU/TensorRT acceleration
    
    Based on RapidLaTeXOCR encoder-decoder architecture
    Extracted to libs for better control and GPU support
    
    Hardware acceleration:
    - TensorRT (best performance)
    - CUDA (good performance)
    - CPU (fallback)
    """
    
    def __init__(self, use_gpu: bool = False, use_tensorrt: bool = False):
        """Initialize formula recognizer with encoder-decoder models
        
        Args:
            use_gpu: Enable CUDA GPU acceleration
            use_tensorrt: Enable TensorRT acceleration (requires TensorRT)
        """
        print(f"Initializing formula recognizer (LaTeX OCR)")
        if use_tensorrt:
            print("  Hardware: TensorRT (requested)")
        elif use_gpu:
            print("  Hardware: CUDA (requested)")
        else:
            print("  Hardware: CPU")
        
        # Initialize LaTeX OCR (models download automatically)
        try:
            self.latex_ocr = LaTeXOCR(
                use_gpu=use_gpu,
                use_tensorrt=use_tensorrt,
            )
            print("  LaTeX OCR initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize LaTeX OCR: {e}")
            print("Formula recognition will be disabled")
            self.latex_ocr = None
    
    def recognize(self, image: Image.Image) -> str:
        """
        Recognize LaTeX formula in image
        
        Args:
            image: PIL Image containing formula
            
        Returns:
            LaTeX formula string
        """
        if self.latex_ocr is None:
            return "[Formula]"
        
        try:
            # Run LaTeX OCR (handles all preprocessing internally)
            latex_str, elapsed_time = self.latex_ocr(image)
            return latex_str
        except Exception as e:
            print(f"Formula recognition error: {e}")
            return "[Formula]"
    
    def recognize_region(self, image: Image.Image, bbox: List[int]) -> str:
        """
        Recognize formula in specific region
        
        Args:
            image: Full PIL Image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            LaTeX formula string
        """
        # Crop region with padding
        x1, y1, x2, y2 = bbox
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Skip very small regions
        if x2 - x1 < 10 or y2 - y1 < 10:
            return "[Formula]"
        
        region = image.crop((x1, y1, x2, y2))
        
        # Recognize formula
        return self.recognize(region)
    
    def crop_region(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """
        Crop a formula region from image with padding
        
        Args:
            image: Full PIL Image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Cropped PIL Image or None if region is too small
        """
        x1, y1, x2, y2 = bbox
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Skip very small regions
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        return image.crop((x1, y1, x2, y2))
    
    def recognize_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Recognize LaTeX formulas in a batch of images
        
        Args:
            images: List of PIL Images containing formulas
            
        Returns:
            List of LaTeX formula strings (same order as input)
        """
        if self.latex_ocr is None:
            return ["[Formula]"] * len(images)
        
        results = []
        for image in images:
            if image is None:
                results.append("[Formula]")
            else:
                try:
                    latex_str, _ = self.latex_ocr(image)
                    results.append(latex_str)
                except Exception as e:
                    print(f"Formula recognition error: {e}")
                    results.append("[Formula]")
        
        return results
    
    def __repr__(self):
        return "FormulaRecognizer(LaTeX OCR with GPU/TensorRT support)"

