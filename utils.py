"""
utils.py - Display Utilities
==============================
Helper functions for rendering the real-time HUD (heads-up display)
overlaid on the video frame, including the status bar, indicator panel,
event log, and flashing alert banner.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple


# ── Colour palette (BGR) ──────────────────────────────────────────────────

_BG_DARK   = (25, 25, 25)
_BG_MID    = (40, 40, 40)
_GREEN     = (0, 210, 0)
_RED       = (0, 0, 220)
_ORANGE    = (0, 165, 255)
_CYAN      = (200, 220, 0)
_GREY      = (150, 150, 150)
_WHITE     = (230, 230, 230)


def draw_dashboard(frame: np.ndarray,
                   stats: dict,
                   event_history: List[Tuple[str, str, str]],
                   motion_in_zone: bool,
                   person_in_zone: bool,
                   avg_flow_mag: float) -> np.ndarray:
    """
    Render the full HUD on top of *frame* (in-place + return).

    Layout:
      • Top bar    — system status + live clock
      • Left panel — detection indicators + session stats
      • Bottom bar — scrolling event log
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top status bar ────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 42), _BG_DARK, -1)

    intruder = motion_in_zone or person_in_zone
    status_text  = "!! INTRUDER DETECTED !!" if intruder else "  SYSTEM ARMED  "
    status_color = _RED if intruder else _GREEN
    cv2.putText(overlay, status_text, (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

    ts = datetime.now().strftime("%Y-%m-%d   %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(overlay, ts, (w - tw - 10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1)

    # ── Left side panel ───────────────────────────────────────────────────
    panel_w = 155
    cv2.rectangle(overlay, (0, 42), (panel_w, h), (18, 18, 18), -1)

    indicators = [
        ("MOTION",  motion_in_zone,       _ORANGE),
        ("PERSON",  person_in_zone,        _RED),
        ("FLOW",    avg_flow_mag > 2.0,   _CYAN),
    ]
    for i, (label, active, col) in enumerate(indicators):
        y = 80 + i * 50
        dot_col = col if active else (70, 70, 70)
        state   = "ACTIVE" if active else "CLEAR"
        cv2.circle(overlay, (18, y), 8, dot_col, -1)
        cv2.putText(overlay, label,  (32, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1)
        cv2.putText(overlay, state,  (32, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1)

    # Flow magnitude bar
    y_flow = 240
    bar_val = min(int(avg_flow_mag / 20 * (panel_w - 20)), panel_w - 20)
    cv2.rectangle(overlay, (10, y_flow), (panel_w - 10, y_flow + 10), (60, 60, 60), -1)
    if bar_val > 0:
        cv2.rectangle(overlay, (10, y_flow), (10 + bar_val, y_flow + 10), _CYAN, -1)
    cv2.putText(overlay, f"Flow: {avg_flow_mag:.1f} px", (10, y_flow + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GREY, 1)

    # Session stats
    cv2.putText(overlay, f"Alerts : {stats.get('total_alerts', 0):04d}",
                (10, y_flow + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 100, 100), 1)
    cv2.putText(overlay, f"Time   : {stats.get('duration_str', '0m 00s')}",
                (10, y_flow + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1)

    # Keyboard hints
    hints = ["Q=Quit", "S=Save HM", "H=Toggle HM", "R=Reset HM"]
    for k, hint in enumerate(hints):
        cv2.putText(overlay, hint, (8, h - 80 + k * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)

    # ── Bottom event log ──────────────────────────────────────────────────
    n = len(event_history)
    if n > 0:
        log_h = n * 20 + 26
        cv2.rectangle(overlay, (panel_w, h - log_h), (w, h), (12, 12, 12), -1)
        cv2.putText(overlay, "— RECENT EVENTS —",
                    (panel_w + 8, h - log_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
        for j, (ts_ev, zones, method) in enumerate(event_history):
            text = f"[{ts_ev}]  {zones}  |  {method}"
            col = (100, 200, 255) if j == n - 1 else (110, 110, 110)
            cv2.putText(overlay, text, (panel_w + 8, h - log_h + 34 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # ── Blend overlay ─────────────────────────────────────────────────────
    cv2.addWeighted(overlay, 0.87, frame, 0.13, 0, frame)
    return frame


def draw_alert_banner(frame: np.ndarray) -> np.ndarray:
    """Flash a large red ALERT banner across the centre of the frame."""
    h, w = frame.shape[:2]
    banner_y1 = h // 2 - 30
    banner_y2 = h // 2 + 32
    cv2.rectangle(frame, (0, banner_y1), (w, banner_y2), (0, 0, 200), -1)
    text = "!! INTRUDER DETECTED IN ZONE !!"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    tx = max(0, (w - tw) // 2)
    ty = banner_y1 + th + (banner_y2 - banner_y1 - th) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    return frame


def draw_warmup_overlay(frame: np.ndarray, progress: int) -> np.ndarray:
    """Show a calibration progress bar while the background model warms up."""
    h, w = frame.shape[:2]
    bar_w = int((progress / 100) * (w - 40))
    cv2.rectangle(frame, (20, h // 2 - 5), (w - 20, h // 2 + 25), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h // 2 - 5), (20 + bar_w, h // 2 + 25), (0, 180, 180), -1)
    cv2.putText(frame,
                f"Calibrating background model...  {progress}%",
                (28, h // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return frame
