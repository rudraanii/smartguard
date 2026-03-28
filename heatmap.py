"""
heatmap.py - Motion Heatmap Generator
=======================================
Accumulates foreground masks over the entire session into a floating-point
accumulator. Applies exponential decay so recent motion weighs more.
The result is mapped to a JET colour map (blue → green → yellow → red).

Useful for post-session analysis: highlights doorways, corridors, or
any area that was repeatedly traversed.

Syllabus Coverage:
  - Image Enhancement / colour mapping      (Topic 1)
  - Histogram processing (via COLORMAP_JET) (Topic 1)
  - Spatio-Temporal Analysis                (Topic 4: Motion Analysis)
"""

import cv2
import numpy as np
import os


class MotionHeatmap:
    """
    Maintains a decaying accumulator of motion activity across frames
    and visualises it as a pseudo-colour heatmap.
    """

    def __init__(self, frame_shape: tuple, decay: float = 0.99):
        """
        Args:
            frame_shape : (height, width[, channels]) — only h×w are used.
            decay       : Per-frame multiplicative decay (0–1).
                          0.99 = very slow fade; 0.90 = fast fade.
        """
        h, w = frame_shape[:2]
        self.accumulator = np.zeros((h, w), dtype=np.float32)
        self.decay = decay
        self.peak = 1.0         # Running maximum for normalisation

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, fg_mask: np.ndarray):
        """
        Add a new foreground mask to the accumulator.

        The mask is normalised to [0, 1] and added after decaying existing
        values, so older motion gradually fades.
        """
        normalised = fg_mask.astype(np.float32) / 255.0
        self.accumulator = self.accumulator * self.decay + normalised
        curr_max = float(self.accumulator.max())
        if curr_max > self.peak:
            self.peak = curr_max

    # ── Visualisation ─────────────────────────────────────────────────────

    def get_heatmap_bgr(self) -> np.ndarray:
        """
        Return the heatmap as a BGR image using the JET colour map.

        Colour scale:
          Blue  (cool)   →  rarely visited
          Green (medium) →  moderately active
          Red   (hot)    →  frequently active area
        """
        if self.peak == 0:
            h, w = self.accumulator.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        normalised = (self.accumulator / self.peak * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blend the heatmap on top of the given frame.

        Args:
            alpha : Heatmap opacity (0 = invisible, 1 = fully opaque).
        """
        heatmap = self.get_heatmap_bgr()
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the current heatmap to a PNG file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        heatmap = self.get_heatmap_bgr()
        cv2.imwrite(path, heatmap)
        print(f"[INFO] Heatmap saved → {path}")

    def reset(self):
        """Clear the accumulator."""
        self.accumulator[:] = 0.0
        self.peak = 1.0
        print("[INFO] Heatmap reset.")
