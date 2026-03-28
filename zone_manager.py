"""
zone_manager.py - Security Zone Manager
=========================================
Handles interactive drawing, storage, and evaluation of polygonal
security zones on the video frame.

Syllabus Coverage:
  - Image segmentation / region-based analysis (Topic 3)
  - Affine / projective concepts via polygon geometry (Topic 1)
"""

import cv2
import numpy as np
from typing import List, Tuple


class ZoneManager:
    """
    Manages user-defined polygonal security zones.
    Supports interactive drawing and real-time intrusion checking.
    """

    def __init__(self):
        self.zones: List[np.ndarray] = []
        self.zone_names: List[str] = []
        self.current_points: List[Tuple[int, int]] = []

    # ── Interactive Zone Setup ────────────────────────────────────────────

    def setup_zones_interactively(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Opens a window where the user can draw polygonal security zones.

        Controls:
          Left Click  → Add a vertex to the current zone
          Right Click → Complete the current zone (min 3 points)
          R           → Reset / discard the in-progress zone
          D           → Delete the last completed zone
          Q / Enter   → Finish setup and begin detection
        """
        ref_frame = frame.copy()
        window = "SmartGuard | Zone Setup  [LClick=Add Pt | RClick=Done Zone | Q=Start]"

        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._complete_zone()

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)

        print("\n[ZONE SETUP MODE]")
        print("  Left Click  → Add vertex to zone")
        print("  Right Click → Complete zone (need ≥3 points)")
        print("  R key       → Reset current in-progress zone")
        print("  D key       → Delete last completed zone")
        print("  Q / Enter   → Done — start detection\n")

        while True:
            display = ref_frame.copy()
            self._draw_completed_zones(display)

            # Draw the zone currently being built
            for pt in self.current_points:
                cv2.circle(display, pt, 5, (0, 255, 255), -1)
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(display, self.current_points[i],
                             self.current_points[i + 1], (0, 255, 255), 2)

            # HUD
            hint = f"Building Zone {len(self.zones)+1}  |  Points: {len(self.current_points)}"
            cv2.putText(display, hint, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            h = display.shape[0]
            cv2.putText(display,
                        "LClick=Add Pt | RClick=Finish Zone | R=Reset | D=Del Last | Q=Done",
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q'), 13):      # Q or Enter
                break
            elif key in (ord('r'), ord('R')):
                self.current_points = []
                print("[INFO] Current zone reset.")
            elif key in (ord('d'), ord('D')):
                if self.zones:
                    removed = self.zone_names.pop()
                    self.zones.pop()
                    print(f"[INFO] Deleted '{removed}'. {len(self.zones)} zone(s) remain.")
                else:
                    print("[INFO] No zones to delete.")
            elif key in (ord('c'), ord('C')):
                self._complete_zone()

        cv2.destroyWindow(window)

        if not self.zones:
            print("[WARN] No zones drawn — defaulting to full frame.")
            self._add_full_frame_zone(frame.shape)

        print(f"[INFO] {len(self.zones)} security zone(s) active.\n")
        return self.zones

    def _complete_zone(self):
        """Finalise the polygon being drawn if it has ≥3 vertices."""
        if len(self.current_points) >= 3:
            poly = np.array(self.current_points, dtype=np.int32)
            self.zones.append(poly)
            name = f"Zone {len(self.zones)}"
            self.zone_names.append(name)
            print(f"[INFO] {name} completed ({len(self.current_points)} vertices).")
            self.current_points = []
        else:
            print("[WARN] Need at least 3 points to close a zone.")

    # ── Default / Preset Zones ────────────────────────────────────────────

    def add_default_zones(self, frame_shape: tuple):
        """Split the frame into left and right security zones."""
        h, w = frame_shape[:2]
        mid = w // 2
        self.zones = [
            np.array([[0, 0], [mid, 0], [mid, h], [0, h]], dtype=np.int32),
            np.array([[mid, 0], [w, 0], [w, h], [mid, h]], dtype=np.int32),
        ]
        self.zone_names = ["Zone 1 (Left)", "Zone 2 (Right)"]
        print("[INFO] Default zones applied (Left / Right halves).")

    def _add_full_frame_zone(self, shape: tuple):
        h, w = shape[:2]
        self.zones.append(
            np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        )
        self.zone_names.append("Zone 1 (Full Frame)")

    # ── Intrusion Checking ────────────────────────────────────────────────

    def check_point_in_zones(self, point: Tuple[int, int]) -> List[int]:
        """Return indices of zones that contain the given (x, y) point."""
        hit = []
        for i, zone in enumerate(self.zones):
            if cv2.pointPolygonTest(zone, (float(point[0]), float(point[1])), False) >= 0:
                hit.append(i)
        return hit

    def check_contour_in_zones(self, contour: np.ndarray) -> List[int]:
        """Return indices of zones overlapping the given motion contour."""
        x, y, w, h = cv2.boundingRect(contour)
        test_points = [
            (x + w // 2, y + h // 2),  # centre
            (x, y), (x + w, y),         # top corners
            (x, y + h), (x + w, y + h), # bottom corners
        ]
        hit = set()
        for pt in test_points:
            for idx in self.check_point_in_zones(pt):
                hit.add(idx)
        return list(hit)

    # ── Drawing ───────────────────────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray, triggered: List[int]) -> np.ndarray:
        """
        Draw all zones on frame with colour coding:
          Green  → zone clear
          Red    → zone breached (intruder detected)
        """
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            alert = i in triggered
            border_col = (0, 0, 220) if alert else (0, 220, 0)
            fill_col   = (0, 0, 120) if alert else (0, 80, 0)

            cv2.fillPoly(overlay, [zone], fill_col)
            cv2.polylines(frame, [zone], True, border_col, 2)

            # Label at centroid
            M = cv2.moments(zone)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = "!! ALERT !!" if alert else self.zone_names[i]
                cv2.putText(frame, label, (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_col, 2)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        return frame

    def _draw_completed_zones(self, frame: np.ndarray):
        """Helper for setup mode — draw finalized zones."""
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            cv2.fillPoly(overlay, [zone], (0, 60, 0))
            cv2.polylines(frame, [zone], True, (0, 200, 0), 2)
            M = cv2.moments(zone)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, self.zone_names[i], (cx - 35, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
