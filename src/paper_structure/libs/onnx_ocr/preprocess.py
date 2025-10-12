"""Preprocessing operations for OCR."""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class DetResizeForTest:
    """Resize image for text detection."""
    
    def __init__(self, limit_side_len=960, limit_type='max', **kwargs):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
    
    def __call__(self, data: Dict) -> Dict:
        img = data['image']
        src_h, src_w, _ = img.shape
        
        if self.limit_type == 'max':
            # Resize so the longer side = limit_side_len
            if max(src_h, src_w) > self.limit_side_len:
                if src_h > src_w:
                    ratio = float(self.limit_side_len) / src_h
                else:
                    ratio = float(self.limit_side_len) / src_w
            else:
                ratio = 1.0
        elif self.limit_type == 'min':
            # Resize so the shorter side = limit_side_len
            if min(src_h, src_w) < self.limit_side_len:
                if src_h < src_w:
                    ratio = float(self.limit_side_len) / src_h
                else:
                    ratio = float(self.limit_side_len) / src_w
            else:
                ratio = 1.0
        else:
            raise ValueError(f"Unknown limit_type: {self.limit_type}")
        
        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)
        
        # Make dimensions divisible by 32
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        img = cv2.resize(img, (resize_w, resize_h))
        
        ratio_h = resize_h / float(src_h)
        ratio_w = resize_w / float(src_w)
        
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data


class NormalizeImage:
    """Normalize image values."""
    
    def __init__(self, scale='1./255.', mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], order='hwc', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale)
        self.mean = np.array(mean).reshape((1, 1, 3)).astype('float32')
        self.std = np.array(std).reshape((1, 1, 3)).astype('float32')
        self.order = order
    
    def __call__(self, data: Dict) -> Dict:
        img = data['image'].astype('float32')
        
        if self.order == 'hwc':
            img = img * self.scale
            img = (img - self.mean) / self.std
        else:  # 'chw'
            img = img * self.scale
            img = (img.transpose(1, 2, 0) - self.mean) / self.std
            img = img.transpose(2, 0, 1)
        
        data['image'] = img
        return data


class ToCHWImage:
    """Convert image from HWC to CHW format."""
    
    def __call__(self, data: Dict) -> Dict:
        img = data['image']
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys:
    """Keep only specified keys in data dict."""
    
    def __init__(self, keep_keys: List[str], **kwargs):
        self.keep_keys = keep_keys
    
    def __call__(self, data: Dict) -> Tuple:
        """Return tuple of (image, shape_list) for detection."""
        result = []
        for key in self.keep_keys:
            result.append(data[key])
        return tuple(result)


def create_operators(op_param_list: List[Dict]):
    """Create preprocessing operators from config list.
    
    Args:
        op_param_list: List of dicts like [{"OpName": {params}}]
        
    Returns:
        List of operator instances
    """
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = globals()[op_name](**param)
        ops.append(op)
    return ops


def transform(data: Dict, ops: List) -> Tuple:
    """Apply preprocessing operators sequentially.
    
    Args:
        data: Dictionary containing 'image' key
        ops: List of operator instances
        
    Returns:
        Tuple of (processed_image, shape_list) or None if error
    """
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data
