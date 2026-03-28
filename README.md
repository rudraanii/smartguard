# smartguard
SmartGuard is a real-time security camera system that watches a live video feed (webcam or file), lets you draw protected zones on screen, and fires an alert the moment it detects a person or significant movement inside those zones. 
Demo
Normal (zones clear)Alert (intruder in zone)Green zone borders, SYSTEM ARMED statusRed zone fill, flashing banner, event logged

Features
FeatureCV TechniqueSyllabus TopicMoving object isolationBackground Subtraction (MOG2)Topic 4 — Motion AnalysisHuman detectionHOG + Linear SVM (sliding window)Topic 3 — Feature ExtractionMotion trackingLucas-Kanade Optical Flow (KLT)Topic 4 — Motion AnalysisActivity visualisationMotion Heatmap (JET colormap)Topic 1 — Image EnhancementNoise reductionMorphological Erosion + DilationTopic 1 — Image ProcessingSmoothingGaussian Blur (pre-subtraction)Topic 1 — Convolution/FilteringZone intrusion checkPolygon point-in-region testTopic 3 — Segmentation

Project Structure
smartguard/
├── main.py             # Entry point — orchestrates the pipeline
├── config.py           # All tunable parameters in one place
├── zone_manager.py     # Interactive zone drawing + intrusion detection
├── bg_subtractor.py    # MOG2 background subtraction module
├── hog_detector.py     # HOG person detection + Non-Max Suppression
├── optical_flow.py     # Lucas-Kanade sparse optical flow tracker
├── heatmap.py          # Decaying motion heatmap accumulator
├── logger.py           # Structured file logging of alert events
├── utils.py            # HUD rendering helpers
├── requirements.txt
├── logs/               # Detection log files (auto-created)
└── outputs/            # Saved heatmaps and recorded video (auto-created)

Setup Instructions
Prerequisites

Python 3.8 or higher
A webcam or a .mp4 / .avi video file
pip (Python package manager)

Step 1 — Clone the repository
bashgit clone https://github.com/<your-username>/smartguard.git
cd smartguard
Step 2 — Create a virtual environment (recommended)
bash# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
Step 3 — Install dependencies
bashpip install -r requirements.txt
Only two packages are needed:
PackagePurposeopencv-pythonAll CV operations (MOG2, HOG, optical flow, display)numpyArray manipulation

Running the Project
Basic run — webcam with default zones
bashpython main.py
Default zones split the frame into a left half and a right half.

Draw your own security zones
bashpython main.py --setup-zones
A zone-drawing window opens before detection starts.
Controls inside the zone window:
ActionKey / MouseAdd a vertex to the current zoneLeft ClickComplete the current zone (need ≥ 3 points)Right ClickReset (discard) the zone being drawnRDelete the last completed zoneDFinish setup and start detectionQ or Enter

Use a video file instead of webcam
bashpython main.py --source path/to/video.mp4

Save the output video
bashpython main.py --save-output
Output is saved to outputs/recorded.avi.

Show a live heatmap window
bashpython main.py --heatmap-window

Disable HOG (faster on low-end hardware)
bashpython main.py --no-hog

All options combined (example)
bashpython main.py --source 0 --setup-zones --save-output --heatmap-window

Keyboard Controls During Detection
KeyActionQQuitSSave current heatmap to outputs/heatmap.pngRReset heatmap accumulatorHToggle the live heatmap window

Configuration
All parameters are in config.py. Key settings:
python"min_contour_area":   900    # Raise to ignore smaller motion blobs
"bg_var_threshold":   40     # Raise to reduce sensitivity
"hog_scale":          1.05   # Increase (e.g. 1.2) for faster HOG
"alert_cooldown_frames": 30  # Min frames between consecutive alerts

Output Files
FileDescriptionlogs/detections.logTimestamped log of every alert eventoutputs/heatmap.pngAccumulated motion heatmap (saved on quit / S key)outputs/recorded.aviAnnotated output video (only with --save-output)

How It Works — Pipeline Overview
  Video Frame
      │
      ▼
  Gaussian Blur  ──────────────────────►  Noise Reduction
      │
      ▼
  MOG2 Background Subtraction  ────────►  Foreground Mask
      │
      ├──► Morphological Ops (erode/dilate)  ──►  Clean Mask
      │         │
      │         ▼
      │    Contour Detection  ─────────────►  Motion Blobs
      │         │
      │         ▼
      │    Zone Membership Check  ─────────►  motion_in_zone
      │
      ├──► HOG Sliding Window (every 5 frames)
      │         │
      │         ▼
      │    NMS ──► Person Bounding Boxes ──►  person_in_zone
      │
      ├──► Lucas-Kanade Optical Flow  ──────►  Flow Vectors + Magnitude
      │
      └──► Heatmap Accumulator  ──────────►  Session Activity Map
                │
                ▼
           Alert Engine
           (if zone breached + cooldown elapsed)
                │
                ├──► Visual Alert Banner
                └──► Log File Entry

Troubleshooting
ProblemSolutionCannot open video source: 0Check webcam connection; try --source 1Many false alertsIncrease "min_contour_area" or "bg_var_threshold" in config.pyHOG detects nothingEnsure there is enough light; try --no-hog and use motion onlySlow FPSUse --no-hog and/or increase "hog_scale" to 1.2 in config.pyBlack screen on startupBackground model is warming up — wait ~3 seconds

Syllabus Topics Covered

Topic 1 — Gaussian filtering, morphological operations, histogram-based
colour mapping (JET), image enhancement
Topic 3 — HOG feature descriptors, edge/gradient computation, image
pyramids (scale-space), object detection, polygon-based segmentation
Topic 4 — Background subtraction and modelling (MOG2), Lucas-Kanade
optical flow, KLT tracking, spatio-temporal motion accumulation
