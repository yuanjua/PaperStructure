"""
Text Detection Module - Stage 1 of OCR Pipeline

Detects text regions in images using DBNet architecture.
Supports batch processing with independent pre/post-processing.
"""

from pathlib import Path
from typing import List, Union
import numpy as np

from .onnx_base import ONNXInferenceBase
from .config import DetectorConfig
from .preprocess import create_operators, transform
from .postprocess import DBPostProcess


class TextDetector:
    """Text detection module with batch processing support.
    
    This is a standalone module that can be used independently.
    Takes batch of images and returns text bounding boxes for each.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: DetectorConfig = None,
    ):
        """Initialize text detector.
        
        Args:
            model_path: Path to detection ONNX model (det.onnx)
            config: Detector configuration (uses defaults if None)
        """
        if config is None:
            config = DetectorConfig()
        
        self.config = config
        
        # Initialize ONNX session with GPU/TensorRT support
        self.session = ONNXInferenceBase(
            model_path,
            use_gpu=config.use_gpu,
            use_tensorrt=config.use_tensorrt,
        )
        
        # Setup preprocessing pipeline
        self.preprocess_ops = create_operators([
            {
                "DetResizeForTest": {
                    "limit_side_len": config.det_limit_side_len,
                    "limit_type": config.det_limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ])
        
        # Setup postprocessing
        self.postprocess_op = DBPostProcess(
            thresh=config.det_db_thresh,
            box_thresh=config.det_db_box_thresh,
            max_candidates=1000,
            unclip_ratio=config.det_db_unclip_ratio,
            use_dilation=config.use_dilation,
            score_mode=config.det_db_score_mode,
            box_type=config.det_box_type,
        )
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """Preprocess single image for detection.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR
            
        Returns:
            Tuple of (processed_image, shape_info) or None if error
        """
        data = {"image": image.copy()}
        result = transform(data, self.preprocess_ops)
        return result
    
    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """Detect text regions in batch of images.
        
        Args:
            images: Single image or list of images (numpy arrays in BGR)
            
        Returns:
            List of bounding box arrays, one per image.
            Each array has shape (N, 4, 2) for N boxes with 4 corner points.
        """
        # Normalize input to list
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = [images]
        
        all_boxes = []
        
        for img in images:
            boxes = self.detect_single(img)
            all_boxes.append(boxes)
        
        return all_boxes
    
    def detect_single(self, image: np.ndarray) -> np.ndarray:
        """Detect text in a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR
            
        Returns:
            Bounding boxes array of shape (N, 4, 2)
        """
        ori_im = image.copy()
        
        # Preprocessing
        result = self.preprocess(image)
        if result is None:
            return np.array([])
        
        img, shape_info = result
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_info, axis=0)
        
        # Run inference
        input_feed = self.session.get_input_feed(img)
        outputs = self.session.run(input_feed)
        
        # Postprocessing
        preds = {"maps": outputs[0]}
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        
        # Filter and clip boxes
        if self.config.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        
        return dt_boxes
    
    def order_points_clockwise(self, pts):
        """Order 4 points in clockwise order."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect
    
    def clip_det_res(self, points, img_height, img_width):
        """Clip points to image boundaries."""
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    def filter_tag_det_res(self, dt_boxes, image_shape):
        """Filter boxes by size and clip to image boundaries."""
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            
            if rect_width <= 3 or rect_height <= 3:
                continue
            
            dt_boxes_new.append(box)
        
        return np.array(dt_boxes_new) if dt_boxes_new else np.array([])
    
    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        """Filter boxes by clipping only (for polygon mode)."""
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        
        return np.array(dt_boxes_new) if dt_boxes_new else np.array([])
