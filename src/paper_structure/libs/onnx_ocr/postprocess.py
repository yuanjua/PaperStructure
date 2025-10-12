"""Postprocessing modules for OCR outputs."""

import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess:
    """Post-processing for DB (Differentiable Binarization) text detection.
    
    Converts probability maps to bounding boxes.
    """
    
    def __init__(
        self,
        thresh=0.3,
        box_thresh=0.7,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        score_mode="fast",
        box_type='quad',
    ):
        """Initialize DB post-processor.
        
        Args:
            thresh: Binarization threshold for probability map
            box_thresh: Minimum confidence score for boxes
            max_candidates: Maximum number of text boxes to detect
            unclip_ratio: Ratio for expanding text regions
            use_dilation: Apply morphological dilation
            score_mode: 'fast' or 'slow' scoring method
            box_type: 'quad' (4 points) or 'poly' (polygon)
        """
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        
        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])
    
    def __call__(self, pred_dict, shape_list):
        """Convert prediction maps to bounding boxes.
        
        Args:
            pred_dict: Dictionary with 'maps' key containing probability maps
            shape_list: Original image shapes [H, W, ratio_h, ratio_w]
            
        Returns:
            List of dictionaries with 'points' key containing boxes
        """
        pred = pred_dict['maps']
        segmentation = pred > self.thresh
        
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel
                )
            else:
                mask = segmentation[batch_index]
            
            if self.box_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            else:
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            
            boxes_batch.append({'points': boxes})
        
        return boxes_batch
    
    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """Extract polygon boxes from binary bitmap."""
        # Ensure bitmap is 2D
        if len(bitmap.shape) == 3:
            bitmap = bitmap.squeeze()
        if len(bitmap.shape) != 2:
            raise ValueError(f"Expected 2D bitmap, got shape {bitmap.shape}")
        
        height, width = bitmap.shape
        boxes = []
        scores = []
        
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            
            if points.shape[0] < 4:
                continue
            
            score = self.box_score_fast(pred, points)
            if score < self.box_thresh:
                continue
            
            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        
        return boxes, scores
    
    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """Extract quad boxes from binary bitmap."""
        # Ensure bitmap is 2D
        if len(bitmap.shape) == 3:
            bitmap = bitmap.squeeze()
        if len(bitmap.shape) != 2:
            raise ValueError(f"Expected 2D bitmap, got shape {bitmap.shape}")
        
        height, width = bitmap.shape
        
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]
        
        num_contours = min(len(contours), self.max_candidates)
        
        boxes = []
        scores = []
        
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            
            if sside < self.min_size:
                continue
            
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            
            if score < self.box_thresh:
                continue
            
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            
            if sside < self.min_size + 2:
                continue
            
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        
        return np.array(boxes, dtype="int32"), scores
    
    def unclip(self, box, unclip_ratio):
        """Expand box using Vatti clipping algorithm."""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    
    def get_mini_boxes(self, contour):
        """Get minimum area rectangle."""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])
    
    def box_score_fast(self, bitmap, box):
        """Calculate box confidence score using bbox mean."""
        # Ensure bitmap is 2D
        if len(bitmap.shape) == 3:
            bitmap = bitmap.squeeze()
        if len(bitmap.shape) != 2:
            raise ValueError(f"Expected 2D bitmap for scoring, got shape {bitmap.shape}")
        
        h, w = bitmap.shape[:2]
        box = box.copy()
        
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class ClsPostProcess:
    """Post-processing for text orientation classification."""
    
    def __init__(self, label_list=None):
        """Initialize classifier post-processor.
        
        Args:
            label_list: List of labels like ['0', '180']
        """
        self.label_list = label_list if label_list else ['0', '180']
    
    def __call__(self, preds):
        """Convert predictions to labels and scores.
        
        Args:
            preds: Prediction probabilities array
            
        Returns:
            List of (label, score) tuples
        """
        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (self.label_list[idx], preds[i, idx])
            for i, idx in enumerate(pred_idxs)
        ]
        return decode_out


class CTCLabelDecode:
    """CTC decoding for text recognition."""
    
    def __init__(self, character_dict_path=None, use_space_char=False):
        """Initialize CTC decoder.
        
        Args:
            character_dict_path: Path to character dictionary file
            use_space_char: Include space character in vocabulary
        """
        self.character_str = []
        
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            
            if use_space_char:
                self.character_str.append(" ")
            
            dict_character = list(self.character_str)
        
        # Add blank token for CTC
        dict_character = ["blank"] + dict_character
        
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
    
    def __call__(self, preds, label=None):
        """Decode CTC predictions to text.
        
        Args:
            preds: Prediction array [batch, time, num_classes]
            label: Optional ground truth labels
            
        Returns:
            List of (text, confidence) tuples
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        
        if label is None:
            return text
        
        label = self.decode(label)
        return text, label
    
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """Convert text indices to strings."""
        result_list = []
        ignored_tokens = [0]  # CTC blank token
        batch_size = len(text_index)
        
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            
            if len(conf_list) == 0:
                conf_list = [0]
            
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        
        return result_list
