"""
Text Orientation Classification Module - Stage 2 of OCR Pipeline

Detects and corrects text orientation (0° or 180°).
Supports batch processing with independent pre/post-processing.
"""

from pathlib import Path
from typing import List, Union, Tuple
import cv2
import numpy as np
import math

from .onnx_base import ONNXInferenceBase
from .config import ClassifierConfig
from .postprocess import ClsPostProcess


class TextClassifier:
    """Text orientation classification module with batch processing.
    
    This is a standalone module that can be used independently.
    Takes batch of text image patches and returns orientation labels.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: ClassifierConfig = None,
    ):
        """Initialize text classifier.
        
        Args:
            model_path: Path to classification ONNX model (cls.onnx)
            config: Classifier configuration (uses defaults if None)
        """
        if config is None:
            config = ClassifierConfig()
        
        self.config = config
        self.cls_image_shape = config.cls_image_shape
        self.cls_batch_num = config.cls_batch_num
        self.cls_thresh = config.cls_thresh
        
        # Initialize ONNX session with GPU/TensorRT support
        self.session = ONNXInferenceBase(
            model_path,
            use_gpu=config.use_gpu,
            use_tensorrt=config.use_tensorrt,
        )
        
        # Setup postprocessing
        self.postprocess_op = ClsPostProcess(label_list=config.label_list)
    
    def resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize image for classification.
        
        Args:
            img: Input image (H, W, C)
            
        Returns:
            Processed image (C, H, W)
        """
        imgC, imgH, imgW = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        
        # Normalize: (x - 0.5) / 0.5
        resized_image -= 0.5
        resized_image /= 0.5
        
        # Pad to fixed width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return padding_im
    
    def __call__(
        self,
        img_list: List[np.ndarray],
        auto_rotate: bool = True
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """Classify and optionally rotate batch of text images.
        
        Args:
            img_list: List of text image patches (BGR format)
            auto_rotate: If True, rotate images classified as 180°
            
        Returns:
            Tuple of:
            - List of (possibly rotated) images
            - List of (label, confidence) tuples
        """
        if not img_list:
            return [], []
        
        img_list = [img.copy() for img in img_list]
        img_num = len(img_list)
        
        # Calculate aspect ratios for sorting (optimization)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        
        cls_res = [["", 0.0]] * img_num
        
        # Process in batches
        for beg_img_no in range(0, img_num, self.cls_batch_num):
            end_img_no = min(img_num, beg_img_no + self.cls_batch_num)
            norm_img_batch = []
            
            # Preprocess batch
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            
            norm_img_batch = np.concatenate(norm_img_batch)
            
            # Run inference
            input_feed = self.session.get_input_feed(norm_img_batch)
            outputs = self.session.run(input_feed)
            prob_out = outputs[0]
            
            # Postprocess
            cls_result = self.postprocess_op(prob_out)
            
            # Store results and rotate if needed
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                original_idx = indices[beg_img_no + rno]
                cls_res[original_idx] = [label, score]
                
                # Rotate if upside-down and confident
                if auto_rotate and "180" in label and score > self.cls_thresh:
                    img_list[original_idx] = cv2.rotate(
                        img_list[original_idx],
                        cv2.ROTATE_180
                    )
        
        return img_list, cls_res
    
    def classify_only(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classify orientation without rotating images.
        
        Args:
            img_list: List of text image patches
            
        Returns:
            List of (label, confidence) tuples
        """
        _, cls_res = self(img_list, auto_rotate=False)
        return cls_res
