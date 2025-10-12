"""Configuration classes for OCR modules."""

from dataclasses import dataclass
from typing import List


@dataclass
class DetectorConfig:
    """Configuration for text detection stage."""
    det_limit_side_len: int = 960  # Maximum side length for input images
    det_limit_type: str = "max"  # 'max' or 'min'
    det_db_thresh: float = 0.3  # Binarization threshold
    det_db_box_thresh: float = 0.6  # Box confidence threshold
    det_db_unclip_ratio: float = 1.5  # Text region expansion ratio
    det_db_score_mode: str = "fast"  # 'fast' or 'slow'
    det_box_type: str = "quad"  # 'quad' or 'poly'
    use_dilation: bool = False  # Apply dilation to binary mask
    use_gpu: bool = False  # Enable CUDA GPU acceleration
    use_tensorrt: bool = False  # Enable TensorRT acceleration


@dataclass
class ClassifierConfig:
    """Configuration for text orientation classification stage."""
    cls_image_shape: List[int] = None  # [C, H, W] e.g., [3, 48, 192]
    cls_batch_num: int = 6  # Batch size for classification
    cls_thresh: float = 0.9  # Confidence threshold for rotation
    label_list: List[str] = None  # e.g., ['0', '180']
    use_gpu: bool = False  # Enable CUDA GPU acceleration
    use_tensorrt: bool = False  # Enable TensorRT acceleration
    
    def __post_init__(self):
        if self.cls_image_shape is None:
            self.cls_image_shape = [3, 48, 192]
        if self.label_list is None:
            self.label_list = ['0', '180']


@dataclass
class RecognizerConfig:
    """Configuration for text recognition stage."""
    rec_image_shape: List[int] = None  # [C, H, W] e.g., [3, 48, 320]
    rec_batch_num: int = 6  # Batch size for recognition
    rec_algorithm: str = "SVTR_LCNet"  # Recognition algorithm
    use_space_char: bool = True  # Include space character in vocabulary
    drop_score: float = 0.5  # Minimum confidence score
    use_gpu: bool = False  # Enable CUDA GPU acceleration
    use_tensorrt: bool = False  # Enable TensorRT acceleration
    
    def __post_init__(self):
        if self.rec_image_shape is None:
            self.rec_image_shape = [3, 48, 320]
