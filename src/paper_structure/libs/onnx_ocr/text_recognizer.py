"""
Text Recognition Module - Stage 3 of OCR Pipeline

Recognizes text from oriented text image patches.
Supports batch processing with independent pre/post-processing.
"""

from pathlib import Path
from typing import List, Union, Tuple
import cv2
import numpy as np
import math

from .onnx_base import ONNXInferenceBase
from .config import RecognizerConfig
from .postprocess import CTCLabelDecode


class TextRecognizer:
    """Text recognition module with batch processing.
    
    This is a standalone module that can be used independently.
    Takes batch of text image patches and returns recognized text strings.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        char_dict_path: Union[str, Path],
        config: RecognizerConfig = None,
    ):
        """Initialize text recognizer.
        
        Args:
            model_path: Path to recognition ONNX model (rec.onnx)
            char_dict_path: Path to character dictionary file
            config: Recognizer configuration (uses defaults if None)
        """
        if config is None:
            config = RecognizerConfig()
        
        self.config = config
        self.rec_image_shape = config.rec_image_shape
        self.rec_batch_num = config.rec_batch_num
        self.rec_algorithm = config.rec_algorithm
        
        # Initialize ONNX session with GPU/TensorRT support
        self.session = ONNXInferenceBase(
            model_path,
            use_gpu=config.use_gpu,
            use_tensorrt=config.use_tensorrt,
        )
        
        # Setup postprocessing (CTC decoder)
        self.postprocess_op = CTCLabelDecode(
            character_dict_path=str(char_dict_path),
            use_space_char=config.use_space_char,
        )
    
    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """Resize and normalize image for recognition.
        
        Args:
            img: Input image (H, W, C) in BGR
            max_wh_ratio: Maximum width/height ratio in batch
            
        Returns:
            Processed image (C, H, W)
        """
        imgC, imgH, imgW = self.rec_image_shape
        
        # Handle different algorithms
        if self.rec_algorithm in ["NRTR", "ViTSTR"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            from PIL import Image
            image_pil = Image.fromarray(np.uint8(img))
            
            if self.rec_algorithm == "ViTSTR":
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.LANCZOS)
            
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            
            if self.rec_algorithm == "ViTSTR":
                norm_img = norm_img.astype(np.float32) / 255.0
            else:
                norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
            
            return norm_img
        
        elif self.rec_algorithm == "RFL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype("float32")
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image
        
        # Default algorithm (SVTR_LCNet, CRNN, etc.)
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        if self.rec_algorithm == "RARE":
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        
        # Pad to fixed width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return padding_im
    
    def __call__(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize text in batch of images.
        
        Args:
            img_list: List of text image patches (BGR format)
            
        Returns:
            List of (text, confidence) tuples
        """
        if not img_list:
            return []
        
        img_num = len(img_list)
        
        # Calculate aspect ratios for sorting (optimization)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        
        rec_res = [["", 0.0]] * img_num
        
        # Process in batches
        for beg_img_no in range(0, img_num, self.rec_batch_num):
            end_img_no = min(img_num, beg_img_no + self.rec_batch_num)
            norm_img_batch = []
            
            # Calculate max aspect ratio in this batch
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            
            # Preprocess batch
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(
                    img_list[indices[ino]],
                    max_wh_ratio
                )
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            
            norm_img_batch = np.concatenate(norm_img_batch)
            
            # Run inference
            input_feed = self.session.get_input_feed(norm_img_batch)
            outputs = self.session.run(input_feed)
            preds = outputs[0]
            
            # Postprocess (CTC decoding)
            rec_result = self.postprocess_op(preds)
            
            # Store results
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        
        return rec_res
    
    def recognize_single(self, img: np.ndarray) -> Tuple[str, float]:
        """Recognize text in a single image.
        
        Args:
            img: Text image patch (BGR format)
            
        Returns:
            Tuple of (text, confidence)
        """
        results = self([img])
        return results[0] if results else ("", 0.0)
