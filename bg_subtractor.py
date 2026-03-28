"""
bg_subtractor.py - Background Subtraction Module
==================================================
Implements MOG2 (Mixture of Gaussians v2) background modelling to
separate moving foreground objects from the static scene.

Morphological operations (erosion + dilation) are applied to suppress
noise and fill holes in the foreground mask.

Syllabus Coverage:
  - Background Subtraction and Modeling       (Topic 4: Motion Analysis)
  - Convolution and Filtering (Gaussian blur) (Topic 1)
  - Image Enhancement via morphological ops   (Topic 1)
  - Histogram Processing (internal to MOG2)   (Topic 1)
"""

import cv2
import numpy as np


class BackgroundSubtractor:
    """
    Wraps OpenCV's MOG2 background subtractor with preprocessing
    and post-processing steps for clean foreground masks.
    """

    def __init__(self, history: int = 500, var_threshold: float = 40,
                 detect_shadows: bool = True, kernel_size: int = 5):
        """
        Args:
            history       : Frames used to model the background.
            var_threshold : Mahalanobis distance threshold — higher means
                            the pixel must change more to be called foreground.
            detect_shadows: MOG2 marks shadows (pixel value 127) separately.
            kernel_size   : Elliptical structuring element size for morphology.
        """
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        # Elliptical kernel: better at preserving rounded shapes (humans)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.frame_count = 0
        self._warmup_frames = 50

    def apply(self, frame: np.ndarray):
        """
        Process one frame through the background subtraction pipeline.

        Pipeline:
          1. Convert to grayscale
          2. Gaussian blur (reduces sensor noise before subtraction)
          3. MOG2 foreground mask
          4. Threshold to remove shadow pixels (value = 127 in MOG2 mask)
          5. Erode → remove tiny noise specks
          6. Dilate → fill holes in large moving objects
          7. Find contours of motion blobs

        Returns:
            fg_mask  (np.ndarray): Binary 8-bit foreground mask (255 = motion)
            contours (list)      : Raw contours from findContours
        """
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply MOG2 — learning rate of -1 lets OpenCV choose automatically
        raw_mask = self.subtractor.apply(blurred, learningRate=-1)

        # Threshold out shadows (127) — keep only definite foreground (255)
        _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return fg_mask, list(contours)

    def get_significant_contours(self, contours: list, min_area: int = 900) -> list:
        """Filter contours to only those exceeding the minimum pixel area."""
        return [c for c in contours if cv2.contourArea(c) >= min_area]

    def is_warmed_up(self) -> bool:
        """True once enough frames have been seen to build a stable model."""
        return self.frame_count >= self._warmup_frames

    def warmup_progress(self) -> int:
        """Return warm-up percentage (0–100)."""
        return min(100, int(self.frame_count / self._warmup_frames * 100))

    def get_background_image(self) -> np.ndarray:
        """Return the current background model as a BGR image."""
        return self.subtractor.getBackgroundImage()
