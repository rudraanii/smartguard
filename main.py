"""
SmartGuard — Intruder Detection via Motion Zones
=================================================
A real-time computer vision security system that monitors user-defined
polygonal zones and raises alerts when motion or a person is detected.

Detection pipeline:
  1. Background Subtraction (MOG2)       — isolate moving objects
  2. HOG Person Detection (sliding window) — confirm human presence
  3. Optical Flow (Lucas-Kanade / KLT)   — track motion direction/speed
  4. Zone checking                       — trigger alert only inside zones
  5. Motion Heatmap                      — accumulate activity over time
  6. Structured logging                  — write events to a log file

Usage:
  python main.py                            # Webcam, default zones
  python main.py --source 0 --setup-zones  # Draw custom zones interactively
  python main.py --source video.mp4        # Use a video file
  python main.py --source 0 --save-output  # Record output video
  python main.py --source 0 --heatmap-window  # Live heatmap window

Keyboard controls during detection:
  Q  — quit
  S  — save heatmap to outputs/heatmap.png
  R  — reset heatmap accumulator
  H  — toggle live heatmap window

Author : [Your Name]
Course : Computer Vision
"""

import cv2
import argparse
import os
import sys

from config import CONFIG
from zone_manager import ZoneManager
from bg_subtractor import BackgroundSubtractor
from hog_detector import HOGPersonDetector
from optical_flow import OpticalFlowTracker
from heatmap import MotionHeatmap
from logger import DetectionLogger
from utils import draw_dashboard, draw_alert_banner, draw_warmup_overlay


# ── CLI Argument Parser ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SmartGuard: Intruder Detection via Motion Zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument(
        "--source", default="0",
        help="Video source: '0' for webcam, or path to a video file (default: 0)"
    )
    p.add_argument(
        "--setup-zones", action="store_true",
        help="Open the interactive zone drawing tool before detection starts"
    )
    p.add_argument(
        "--no-hog", action="store_true",
        help="Disable HOG person detection (speeds up processing)"
    )
    p.add_argument(
        "--no-flow", action="store_true",
        help="Disable optical flow tracking"
    )
    p.add_argument(
        "--save-output", action="store_true",
        help="Record the annotated output stream to outputs/recorded.avi"
    )
    p.add_argument(
        "--heatmap-window", action="store_true",
        help="Display a live motion heatmap in a separate window"
    )
    return p


# ── Video Source Helpers ──────────────────────────────────────────────────

def open_source(source_str: str) -> cv2.VideoCapture:
    """Parse the source argument and return an opened VideoCapture."""
    source_str = source_str.strip()
    try:
        src = int(source_str)
    except ValueError:
        src = source_str
        if not os.path.exists(src):
            print(f"[ERROR] File not found: {src}")
            sys.exit(1)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {src}")
        sys.exit(1)

    label = "Webcam" if isinstance(src, int) else src
    print(f"[INFO] Video source opened: {label}")
    return cap


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    # ── Open video source ─────────────────────────────────────────────────
    cap = open_source(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from video source.")
        sys.exit(1)

    W = CONFIG["frame_width"]
    H = CONFIG["frame_height"]
    first_frame = cv2.resize(first_frame, (W, H))

    # ── Zone setup ────────────────────────────────────────────────────────
    zm = ZoneManager()
    if args.setup_zones:
        print("[INFO] Entering zone setup — draw your security zones.")
        zm.setup_zones_interactively(first_frame)
    else:
        zm.add_default_zones(first_frame.shape)
        print("[INFO] Tip: run with --setup-zones to draw custom zones.")

    # ── Initialise CV modules ─────────────────────────────────────────────
    bg_sub = BackgroundSubtractor(
        history=CONFIG["bg_history"],
        var_threshold=CONFIG["bg_var_threshold"],
        detect_shadows=CONFIG["detect_shadows"],
        kernel_size=CONFIG["morph_kernel_size"]
    )

    hog = (HOGPersonDetector(
        win_stride=CONFIG["hog_win_stride"],
        padding=CONFIG["hog_padding"],
        scale=CONFIG["hog_scale"]
    ) if not args.no_hog else None)

    flow = (OpticalFlowTracker(
        max_corners=CONFIG["lk_max_corners"],
        quality_level=CONFIG["lk_quality_level"],
        min_distance=CONFIG["lk_min_distance"],
        win_size=CONFIG["lk_win_size"],
        max_level=CONFIG["lk_max_level"]
    ) if not args.no_flow else None)

    heatmap = MotionHeatmap((H, W), decay=0.99)
    log = DetectionLogger(CONFIG["log_file"])

    # ── Optional output writer ────────────────────────────────────────────
    writer = None
    if args.save_output:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(CONFIG["output_video"], fourcc, 20, (W, H))
        print(f"[INFO] Recording to: {CONFIG['output_video']}")

    # ── State ─────────────────────────────────────────────────────────────
    frame_num         = 0
    hog_detections    = []
    avg_flow_mag      = 0.0
    alert_cooldown    = 0
    alert_banner_timer = 0
    show_heatmap_win  = args.heatmap_window

    HOG_EVERY = 5   # Run HOG every N frames (computationally expensive)

    print("\n[SmartGuard] Detection active.")
    print("  Q=Quit  S=Save heatmap  R=Reset heatmap  H=Toggle heatmap window\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame = cv2.resize(frame, (W, H))
        frame_num += 1
        warmed = bg_sub.is_warmed_up()

        # ── 1. Background subtraction ─────────────────────────────────────
        fg_mask, raw_contours = bg_sub.apply(frame)
        significant = bg_sub.get_significant_contours(
            raw_contours, CONFIG["min_contour_area"]
        )

        # ── 2. Heatmap accumulation ───────────────────────────────────────
        heatmap.update(fg_mask)

        # ── 3. Optical flow ───────────────────────────────────────────────
        if flow and warmed:
            flow_mask = fg_mask if significant else None
            frame, avg_flow_mag = flow.update(frame, mask=flow_mask)

        # ── 4. HOG person detection (run every HOG_EVERY frames) ──────────
        if hog and warmed and frame_num % HOG_EVERY == 0:
            hog_detections = hog.detect(frame) if significant else []

        if hog and hog_detections:
            frame = hog.draw_detections(frame, hog_detections)

        # ── 5. Zone intrusion check ───────────────────────────────────────
        triggered_zones: set = set()
        motion_in_zone = False
        person_in_zone = False

        # Draw motion contour bounding boxes and check zone membership
        for cnt in significant:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 140, 255), 1)
            for zi in zm.check_contour_in_zones(cnt):
                triggered_zones.add(zi)
                motion_in_zone = True

        # Check HOG detections against zones
        for (px, py, pw, ph) in hog_detections:
            centre = (px + pw // 2, py + ph // 2)
            for zi in zm.check_point_in_zones(centre):
                triggered_zones.add(zi)
                person_in_zone = True

        # ── 6. Draw zones ─────────────────────────────────────────────────
        frame = zm.draw_zones(frame, list(triggered_zones))

        # ── 7. Alert handling ─────────────────────────────────────────────
        intruder = motion_in_zone or person_in_zone

        if intruder and alert_cooldown == 0 and warmed:
            method = (
                "HOG+Motion" if person_in_zone and motion_in_zone else
                "HOG"        if person_in_zone else
                "Motion"
            )
            names = [zm.zone_names[i] for i in triggered_zones]
            log.log_intrusion(names, method)
            alert_cooldown     = CONFIG["alert_cooldown_frames"]
            alert_banner_timer = CONFIG["alert_duration_frames"]

        if alert_cooldown    > 0: alert_cooldown    -= 1
        if alert_banner_timer > 0:
            draw_alert_banner(frame)
            alert_banner_timer -= 1

        # ── 8. Warmup overlay ─────────────────────────────────────────────
        if not warmed:
            draw_warmup_overlay(frame, bg_sub.warmup_progress())

        # ── 9. HUD ───────────────────────────────────────────────────────
        frame = draw_dashboard(
            frame,
            log.get_session_stats(),
            log.get_event_history(),
            motion_in_zone,
            person_in_zone,
            avg_flow_mag
        )

        # ── 10. Optional heatmap window ───────────────────────────────────
        if show_heatmap_win:
            hm_small = cv2.resize(heatmap.get_heatmap_bgr(), (W // 2, H // 2))
            cv2.imshow("SmartGuard | Motion Heatmap", hm_small)

        # ── Display + write ───────────────────────────────────────────────
        cv2.imshow("SmartGuard | Intruder Detection", frame)
        if writer:
            writer.write(frame)

        # ── Keyboard controls ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            heatmap.save(CONFIG["heatmap_output"])
        elif key in (ord('r'), ord('R')):
            heatmap.reset()
        elif key in (ord('h'), ord('H')):
            show_heatmap_win = not show_heatmap_win
            if not show_heatmap_win:
                cv2.destroyWindow("SmartGuard | Motion Heatmap")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n[INFO] Shutting down SmartGuard...")
    heatmap.save(CONFIG["heatmap_output"])
    log.close()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stats = log.get_session_stats()
    print(f"[INFO] Session ended | Duration: {stats['duration_str']} | "
          f"Total alerts: {stats['total_alerts']}")
    print(f"[INFO] Log file  → {CONFIG['log_file']}")
    print(f"[INFO] Heatmap   → {CONFIG['heatmap_output']}")


if __name__ == "__main__":
    main()
