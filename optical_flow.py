"""
optical_flow.py - Optical Flow Tracker (Lucas-Kanade / KLT)
=============================================================
Tracks sparse feature points across frames using the
Lucas-Kanade pyramidal optical flow algorithm — also known as
the Kanade-Lucas-Tomasi (KLT) tracker.

Steps:
  1. Detect good corner features using Shi-Tomasi corner detection
  2. For each feature, solve the Lucas-Kanade constraint equations
     in a local neighbourhood (captured by the search window)
  3. A Gaussian image pyramid handles large displacements
  4. Draw arrow vectors showing each point's motion between frames

Syllabus Coverage:
  - Optical Flow                                (Topic 4: Motion Analysis)
  - KLT Tracker                                 (Topic 4)
  - Corner detection / Shi-Tomasi              (Topic 3: Feature Extraction)
  - Scale-Space / Image Pyramids               (Topic 3)
  - Spatio-Temporal Analysis                   (Topic 4)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class OpticalFlowTracker:
    """
    Sparse Lucas-Kanade optical flow tracker.
    Re-detects corner features periodically to maintain stable tracking.
    """

    def __init__(self, max_corners: int = 100, quality_level: float = 0.3,
                 min_distance: int = 7, win_size: Tuple = (15, 15),
                 max_level: int = 2, refresh_every: int = 10):
        """
        Args:
            max_corners   : Max feature points to detect per frame.
            quality_level : Shi-Tomasi quality threshold (0–1).
            min_distance  : Min pixel separation between features.
            win_size      : LK search window per pyramid level.
            max_level     : Number of pyramid levels (0 = no pyramid).
            refresh_every : Re-detect features every N frames.
        """
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
            )
        )
        self.refresh_every = refresh_every
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.frame_idx: int = 0

    def update(self, frame: np.ndarray,
               mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Track features from the previous frame to the current one.

        Args:
            frame : Current BGR video frame.
            mask  : Optional binary mask restricting where features are detected
                    (e.g., the foreground mask from background subtraction).

        Returns:
            frame         : Annotated frame with flow arrows drawn.
            avg_magnitude : Mean displacement (pixels/frame) across all tracked
                            points — a proxy for scene motion intensity.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_idx += 1
        avg_magnitude = 0.0

        # ── Initialise or re-detect features ─────────────────────────────
        if (self.prev_gray is None
                or self.prev_pts is None
                or len(self.prev_pts) < 10
                or self.frame_idx % self.refresh_every == 0):

            detection_mask = mask if mask is not None else None
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=detection_mask, **self.feature_params
            )
            if self.prev_gray is None:
                self.prev_gray = gray
                return frame, avg_magnitude

        # ── Lucas-Kanade optical flow ─────────────────────────────────────
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray,
                self.prev_pts, None,
                **self.lk_params
            )

            if curr_pts is not None and status is not None:
                good_new = curr_pts[status == 1]
                good_old = self.prev_pts[status == 1]

                if len(good_new) > 0:
                    # Compute motion vectors
                    deltas = good_new - good_old
                    magnitudes = np.linalg.norm(deltas, axis=1)
                    avg_magnitude = float(np.mean(magnitudes))

                    # Draw arrows only for points with noticeable movement
                    for new_pt, old_pt, mag in zip(good_new, good_old, magnitudes):
                        if mag > 1.5:
                            a, b = int(new_pt[0]), int(new_pt[1])
                            c, d = int(old_pt[0]), int(old_pt[1])
                            cv2.arrowedLine(
                                frame, (c, d), (a, b),
                                (0, 240, 240), 1, tipLength=0.4
                            )
                            cv2.circle(frame, (a, b), 2, (0, 200, 200), -1)

                self.prev_pts = (
                    good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
                )

        self.prev_gray = gray
        return frame, avg_magnitude
