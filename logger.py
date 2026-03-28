"""
logger.py - Detection Event Logger
=====================================
Handles structured file-based logging of intrusion events and
maintains a short in-memory ring buffer for the on-screen event log.
"""

import os
import logging
from datetime import datetime
from collections import deque
from typing import List, Tuple


class DetectionLogger:
    """
    Logs intrusion events to a timestamped log file and keeps a
    rolling window of recent events for the HUD display.
    """

    def __init__(self, log_file: str = "logs/detections.log",
                 max_history: int = 6):
        """
        Args:
            log_file    : Path to the output log file (created if absent).
            max_history : Number of recent events shown in the HUD.
        """
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self._log_file = log_file
        self._history: deque = deque(maxlen=max_history)
        self._total_alerts = 0
        self._session_start = datetime.now()

        # File logger (separate from root logger to avoid conflicts)
        self._logger = logging.getLogger("SmartGuard")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self._logger.addHandler(fh)

        self._logger.info("=" * 55)
        self._logger.info("SmartGuard session started")
        self._logger.info("=" * 55)
        print(f"[INFO] Logging to: {log_file}")

    # ── Logging helpers ───────────────────────────────────────────────────

    def log_intrusion(self, zone_names: List[str], method: str = "Motion"):
        """
        Record an intrusion event.

        Args:
            zone_names : Names of zones where intrusion was detected.
            method     : Detection method (e.g., 'Motion', 'HOG', 'HOG+Motion').
        """
        self._total_alerts += 1
        zones_str = ", ".join(zone_names) if zone_names else "Unknown"
        ts = datetime.now().strftime("%H:%M:%S")
        msg = (f"ALERT #{self._total_alerts:04d} | "
               f"Zone(s): {zones_str} | Method: {method}")

        self._logger.warning(msg)
        self._history.append((ts, zones_str, method))
        print(f"[ALERT] {ts} — {msg}")

    def log_info(self, message: str):
        """Write a general info message to the log file."""
        self._logger.info(message)

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_event_history(self) -> List[Tuple[str, str, str]]:
        """Return recent events as list of (timestamp, zones, method)."""
        return list(self._history)

    def get_session_stats(self) -> dict:
        """Return a dict with total_alerts and session duration."""
        elapsed = int((datetime.now() - self._session_start).total_seconds())
        mins, secs = divmod(elapsed, 60)
        return {
            "total_alerts": self._total_alerts,
            "duration_str": f"{mins}m {secs:02d}s",
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        """Finalise the log file with session summary."""
        stats = self.get_session_stats()
        self._logger.info(
            f"Session ended | Duration: {stats['duration_str']} | "
            f"Total alerts: {stats['total_alerts']}"
        )
        self._logger.info("=" * 55)
