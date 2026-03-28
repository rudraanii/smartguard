"""
config.py - SmartGuard Configuration
=====================================
Central configuration file for the SmartGuard Intruder Detection System.
Modify these values to tune detection sensitivity and display options.
"""

CONFIG = {
    # ── Video Source ──────────────────────────────────────────────────────
    # 0 = webcam, or set to a path like "sample.mp4" for a video file
    "video_source": 0,
    "frame_width": 640,
    "frame_height": 480,

    # ── Background Subtraction (MOG2) ─────────────────────────────────────
    # history: Number of frames used to build the background model
    # var_threshold: Higher = less sensitive to small changes
    # detect_shadows: True = shadows are marked separately (gray in mask)
    "bg_history": 500,
    "bg_var_threshold": 40,
    "detect_shadows": True,

    # ── Motion Detection ──────────────────────────────────────────────────
    # min_contour_area: Minimum pixel area to qualify as a motion event
    # morph_kernel_size: Size of kernel for noise cleanup (erosion/dilation)
    "min_contour_area": 900,
    "morph_kernel_size": 5,

    # ── HOG Person Detection ──────────────────────────────────────────────
    # win_stride: Sliding window stride — (8,8) is accurate, (16,16) is faster
    # hog_scale: Scale factor between pyramid levels (1.05 = fine, 1.2 = fast)
    "hog_win_stride": (8, 8),
    "hog_padding": (4, 4),
    "hog_scale": 1.05,

    # ── Optical Flow (Lucas-Kanade / KLT) ────────────────────────────────
    "lk_max_corners": 100,
    "lk_quality_level": 0.3,
    "lk_min_distance": 7,
    "lk_win_size": (15, 15),
    "lk_max_level": 2,

    # ── Alert System ──────────────────────────────────────────────────────
    # alert_cooldown_frames: Frames to wait between consecutive alert logs
    # alert_duration_frames: Frames to display the banner on screen
    "alert_cooldown_frames": 30,
    "alert_duration_frames": 20,

    # ── Output Paths ──────────────────────────────────────────────────────
    "log_file": "logs/detections.log",
    "output_dir": "outputs",
    "output_video": "outputs/recorded.avi",
    "heatmap_output": "outputs/heatmap.png",
}
