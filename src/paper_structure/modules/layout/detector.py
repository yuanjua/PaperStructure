"""
Layout Detection Module
Uses YOLOX for document layout analysis with hardware acceleration
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any
from paper_structure.libs.yolox import get_model


class LayoutDetector:
    """
    Layout detection using YOLOX ONNX model
    
    Hardware acceleration:
    - Automatically uses available providers (TensorRT > CUDA > CPU)
    - Configured by our custom YOLOX library (extracted from unstructured-inference)
    """
    
    def __init__(self, model_name: str = "yolox"):
        """
        Initialize layout detector
        
        Args:
            model_name: YOLOX model variant (yolox, yolox_tiny, yolox_quantized)
        """
        self.model_name = model_name
        # YOLOX library automatically selects best available provider
        # Priority: TensorrtExecutionProvider > CUDAExecutionProvider > CPUExecutionProvider
        self.model = get_model(model_name)
        
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect layout elements in image
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected elements with bbox, type, and confidence
        """
        # Run YOLOX detection
        layout_elements = self.model.predict(image)
        elements_list = layout_elements.as_list()
        
        # Convert to standard format
        results = []
        for element in elements_list:
            bbox = element.bbox
            result = {
                'bbox': [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)],
                'type': element.type if hasattr(element, 'type') else 'Unknown',
                'confidence': float(element.prob) if hasattr(element, 'prob') else 0.0,
                'element': element  # Keep original for reference
            }
            results.append(result)
        
        return results
    
    def __repr__(self):
        return f"LayoutDetector(model={self.model_name})"
