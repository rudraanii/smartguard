"""
hog_detector.py - HOG Person Detector
========================================
Uses Histogram of Oriented Gradients (HOG) descriptors combined with
a pre-trained linear SVM (bundled in OpenCV) to detect human figures.

HOG captures the local gradient structure of image patches by:
  1. Computing image gradients (magnitude + orientation)
  2. Binning gradients into orientation histograms (9 bins / 0°–180°)
  3. Grouping cells into blocks and L2-normalising (handles illumination)
  4. Flattening descriptors and scoring with the SVM

A sliding-window + image pyramid approach handles multi-scale detection.
Non-Maximum Suppression (NMS) removes duplicate bounding boxes.

Syllabus Coverage:
  - HOG (Histogram of Oriented Gradients)  (Topic 3: Feature Extraction)
  - Edge / gradient detection concepts     (Topic 3)
  - Scale-space / image pyramids           (Topic 3)
  - Object detection                       (Topic 3)
"""

import cv2
import numpy as np
from typing import List, Tuple


class HOGPersonDetector:
    """
    Wraps OpenCV's HOGDescriptor with the default pedestrian SVM.
    Includes NMS to clean up overlapping detections.
    """

    def __init__(self, win_stride: Tuple = (8, 8), padding: Tuple = (4, 4),
                 scale: float = 1.05, hit_threshold: float = 0.0):
        """
        Args:
            win_stride    : Step between sliding windows — smaller = denser scan.
            padding       : Extra padding around each detection window.
            scale         : Pyramid scale factor (1.05 = fine, 1.2 = fast).
            hit_threshold : SVM decision boundary margin — raise to reduce FPs.
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold

        # HOG is slow — only run it on a downscaled frame
        self._detect_width = 320
        self._detect_height = 240

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in the frame using HOG + sliding window.

        Returns:
            List of (x, y, w, h) bounding boxes in the original frame's
            coordinate space.
        """
        orig_h, orig_w = frame.shape[:2]
        small = cv2.resize(frame, (self._detect_width, self._detect_height))

        scale_x = orig_w / self._detect_width
        scale_y = orig_h / self._detect_height

        rects, _ = self.hog.detectMultiScale(
            small,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )

        if len(rects) == 0:
            return []

        # Scale detections back to original resolution
        boxes = [
            (int(x * scale_x), int(y * scale_y),
             int(w * scale_x), int(h * scale_y))
            for (x, y, w, h) in rects
        ]

        return self._nms(boxes, overlap_thresh=0.65)

    def _nms(self, boxes: List[Tuple], overlap_thresh: float = 0.65) -> List[Tuple]:
        """
        Non-Maximum Suppression: remove overlapping bounding boxes,
        keeping only the one with the lowest bottom edge (best candidate).
        """
        if not boxes:
            return []

        arr = np.array(
            [[x, y, x + w, y + h] for (x, y, w, h) in boxes],
            dtype=float
        )
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        pick = []

        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            inter_w = np.maximum(0, xx2 - xx1 + 1)
            inter_h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (inter_w * inter_h) / areas[idxs[:-1]]
            idxs = np.delete(
                idxs, np.concatenate(([len(idxs) - 1],
                                       np.where(overlap > overlap_thresh)[0]))
            )

        return [boxes[i] for i in pick]

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Tuple]) -> np.ndarray:
        """Annotate frame with HOG detection boxes and labels."""
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 220), 2)
            label_bg_pt1 = (x, y - 20)
            label_bg_pt2 = (x + 80, y)
            cv2.rectangle(frame, label_bg_pt1, label_bg_pt2, (0, 0, 220), -1)
            cv2.putText(frame, "PERSON", (x + 4, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
