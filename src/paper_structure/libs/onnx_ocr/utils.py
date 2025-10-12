"""Utility functions for OCR pipeline."""

import cv2
import numpy as np
from typing import Tuple


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and rotate text region from image.
    
    Args:
        img: Source image
        points: Text region points (4x2 array)
        
    Returns:
        Cropped and rotated text image
    """
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        )
    )
    
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
    ])
    
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )
    
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    
    return dst_img


def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
    """Sort text boxes from top to bottom, left to right.
    
    Args:
        dt_boxes: Detection boxes array (N, 4, 2)
        
    Returns:
        Sorted boxes array
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
               (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    
    return np.array(_boxes)


def draw_ocr_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    texts: list = None,
    scores: list = None,
    drop_score: float = 0.5,
    font_path: str = None
) -> np.ndarray:
    """Draw OCR results on image.
    
    Args:
        image: Source image
        boxes: Detection boxes
        texts: Recognized texts
        scores: Confidence scores
        drop_score: Minimum score threshold
        font_path: Path to font file for text rendering
        
    Returns:
        Image with drawn boxes and text
    """
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, 18)
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    for idx, box in enumerate(boxes):
        if scores and scores[idx] < drop_score:
            continue
        
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        draw.polygon([tuple(p) for p in box], outline=(0, 255, 0))
        
        if texts:
            box_height = int(np.linalg.norm(box[0] - box[3]))
            box_width = int(np.linalg.norm(box[0] - box[1]))
            
            if box_height > 2 * box_width:
                # Vertical text
                pass
            else:
                # Horizontal text
                text = texts[idx] if idx < len(texts) else ""
                draw.text((box[0][0], box[0][1] - 20), text, fill=(255, 0, 0), font=font)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
