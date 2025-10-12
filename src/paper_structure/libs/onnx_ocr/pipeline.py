"""
High-level OCR Pipeline
Combines the three modular stages into a simple interface similar to ONNXPaddleOcr
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .text_detector import TextDetector
from .text_classifier import TextClassifier
from .text_recognizer import TextRecognizer
from .config import DetectorConfig, ClassifierConfig, RecognizerConfig
from .utils import get_rotate_crop_image, sorted_boxes


class OCRPipeline:
    """
    Complete OCR pipeline combining detection, classification, and recognition.
    
    This provides a simple interface similar to ONNXPaddleOcr but using the
    modularized libs implementation.
    
    Usage:
        ocr = OCRPipeline(use_angle_cls=True, use_gpu=False)
        result = ocr.ocr(image)
    """
    
    def __init__(
        self,
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        use_dml: bool = False,
        det_model_path: Optional[str] = None,
        cls_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        char_dict_path: Optional[str] = None,
    ):
        """
        Initialize OCR pipeline
        
        Args:
            use_angle_cls: Whether to use angle classification (rotation correction)
            use_gpu: Enable CUDA GPU acceleration
            use_dml: Enable DirectML acceleration (Windows only, not implemented)
            det_model_path: Path to detection model (default: bundled model)
            cls_model_path: Path to classification model (default: bundled model)
            rec_model_path: Path to recognition model (default: bundled model)
            char_dict_path: Path to character dictionary (default: bundled dict)
        """
        self.use_angle_cls = use_angle_cls
        
        # Get default model paths if not provided
        models_dir = Path(__file__).parent / "models"
        
        if det_model_path is None:
            det_model_path = models_dir / "det" / "det.onnx"
        if cls_model_path is None:
            cls_model_path = models_dir / "cls" / "cls.onnx"
        if rec_model_path is None:
            rec_model_path = models_dir / "rec" / "rec.onnx"
        if char_dict_path is None:
            char_dict_path = models_dir / "ppocrv5_dict.txt"
        
        # Validate model files exist
        for path, name in [
            (det_model_path, "detection model"),
            (rec_model_path, "recognition model"),
            (char_dict_path, "character dictionary"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required {name} not found at: {path}")
        
        if use_angle_cls and not Path(cls_model_path).exists():
            raise FileNotFoundError(f"Classification model not found at: {cls_model_path}")
        
        # Initialize detector
        det_config = DetectorConfig(
            det_limit_side_len=960,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            use_gpu=use_gpu,
            use_tensorrt=False,  # Can be enabled if TensorRT is available
        )
        self.text_detector = TextDetector(det_model_path, det_config)
        
        # Initialize classifier if enabled
        if use_angle_cls:
            cls_config = ClassifierConfig(
                cls_image_shape=[3, 48, 192],
                cls_batch_num=6,
                cls_thresh=0.9,
                use_gpu=use_gpu,
                use_tensorrt=False,
            )
            self.text_classifier = TextClassifier(cls_model_path, cls_config)
        else:
            self.text_classifier = None
        
        # Initialize recognizer
        rec_config = RecognizerConfig(
            rec_image_shape=[3, 48, 320],
            rec_batch_num=6,
            rec_algorithm="SVTR_LCNet",
            use_space_char=True,
            drop_score=0.5,
            use_gpu=use_gpu,
            use_tensorrt=False,
        )
        self.text_recognizer = TextRecognizer(rec_model_path, char_dict_path, rec_config)
    
    def ocr(
        self,
        img: np.ndarray,
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
    ) -> List[List[Tuple]]:
        """
        Perform OCR on image
        
        Args:
            img: Input image as numpy array (BGR format)
            det: Whether to perform detection
            rec: Whether to perform recognition
            cls: Whether to perform angle classification
            
        Returns:
            List containing one element (for compatibility):
            [
                [
                    (box_coordinates, (text, confidence)),
                    ...
                ]
            ]
            
            Where box_coordinates is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if cls and self.use_angle_cls == False:
            print(
                "Since the angle classifier is not initialized, "
                "the angle classifier will not be used during the forward process"
            )
            cls = False
        
        if det and rec:
            # Full pipeline: detect → classify → recognize
            dt_boxes, rec_res = self.__call__(img, cls)
            # Format results to match ONNXPaddleOcr output
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            return [tmp_res]
        
        elif det and not rec:
            # Detection only
            dt_boxes = self.text_detector.detect_single(img)
            dt_boxes = sorted_boxes(dt_boxes)
            tmp_res = [box.tolist() for box in dt_boxes]
            return [tmp_res]
        
        else:
            # Recognition only (assume img is already a text patch)
            if not isinstance(img, list):
                img = [img]
            
            if self.use_angle_cls and cls and self.text_classifier is not None:
                img, _ = self.text_classifier(img, auto_rotate=True)
            
            rec_res = self.text_recognizer(img)
            return [rec_res]
    
    def __call__(self, img: np.ndarray, cls: bool = True) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Internal method for full OCR pipeline
        
        Args:
            img: Input image (BGR)
            cls: Whether to use angle classification
            
        Returns:
            Tuple of (boxes, recognition_results)
            - boxes: numpy array of shape (N, 4, 2)
            - recognition_results: list of (text, confidence) tuples
        """
        # Stage 1: Detect text regions
        dt_boxes = self.text_detector.detect_single(img)
        
        if len(dt_boxes) == 0:
            return np.array([]), []
        
        # Sort boxes top-to-bottom, left-to-right
        dt_boxes = sorted_boxes(dt_boxes)
        
        # Stage 2: Crop text patches
        img_crop_list = []
        for box in dt_boxes:
            tmp_box = box.copy()
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        
        # Stage 3: Classify orientation (optional)
        if self.use_angle_cls and cls and self.text_classifier is not None:
            img_crop_list, _ = self.text_classifier(img_crop_list, auto_rotate=True)
        
        # Stage 4: Recognize text
        rec_res = self.text_recognizer(img_crop_list)
        
        return dt_boxes, rec_res
    
    def __repr__(self):
        return (
            f"OCRPipeline(\n"
            f"  detector={self.text_detector},\n"
            f"  classifier={self.text_classifier},\n"
            f"  recognizer={self.text_recognizer}\n"
            f")"
        )


# Alias for compatibility with ONNXPaddleOcr naming
ONNXPaddleOcr = OCRPipeline
