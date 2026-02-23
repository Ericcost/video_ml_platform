"""
video_processor.py
------------------
Orchestrates the full pipeline:
  1. Read video frame by frame
  2. Detect ball + players (YOLOv8)
  3. Track across frames (IoU tracker)
  4. Classify action (heuristics)
  5. Write annotated output video
  6. Return structured result
"""

import cv2
import io
import logging
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from detector import VolleyballDetector
from tracker import SimpleTracker
from action_classifier import ActionClassifier, EventSegmenter

log = logging.getLogger(__name__)

# Annotation colors (BGR)
COLORS = {
    "team_a":  (220, 80,  30),
    "team_b":  (30,  100, 220),
    "unknown": (160, 160, 160),
    "ball":    (0,   240, 240),
    "action":  (255, 255,  50),
}

ACTION_LABELS = {
    "serve":  "SERVE 🏐",
    "pass":   "PASS",
    "set":    "SET",
    "attack": "ATTACK ⚡",
    "block":  "BLOCK 🚧",
    "dig":    "DIG",
    "other":  "",
}


def _draw_annotations(
    frame: np.ndarray,
    detections,
    action_label: str,
    frame_number: int,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw players
    for player in detections.players:
        d    = player.detection
        color = COLORS.get(player.team, COLORS["unknown"])
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"#{player.track_id}"
        if player.jersey_number:
            label += f" {player.jersey_number}"
        if player.team != "unknown":
            label += f" ({player.team[-1].upper()})"

        cv2.rectangle(out, (x1, y1-20), (x1+len(label)*8, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw ball
    if detections.ball:
        b  = detections.ball
        cx, cy = int(b.cx), int(b.cy)
        r  = max(8, int(min(b.w, b.h) / 2))
        cv2.circle(out, (cx, cy), r, COLORS["ball"], 2)
        cv2.circle(out, (cx, cy), 3, COLORS["ball"], -1)

    # Action overlay
    if action_label and action_label != ACTION_LABELS["other"]:
        cv2.rectangle(out, (10, h-55), (350, h-10), (0, 0, 0), -1)
        cv2.putText(out, action_label, (18, h-20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, COLORS["action"], 2, cv2.LINE_AA)

    # Frame counter
    ts = frame_number / max(fps, 1)
    cv2.putText(out, f"{int(ts//60):02d}:{ts%60:05.2f}", (w-130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    frame_skip: int = 2,       # process every N frames (speed vs accuracy)
) -> dict:
    """
    Full analysis pipeline for a single video file.

    Args:
        input_path:        path to input video
        output_path:       path for annotated output video
        progress_callback: called with float 0-1 as processing advances
        frame_skip:        process 1 frame every N (2 = process at half FPS)

    Returns:
        result dict matching AnalysisResult schema
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = total_frames / fps

    log.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")

    # ── Setup ──────────────────────────────────────────────────────────────
    detector   = VolleyballDetector(model_path="yolov8n.pt", device="cpu")
    tracker    = SimpleTracker()
    classifier = ActionClassifier(frame_width=width, frame_height=height)
    segmenter  = EventSegmenter()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number    = 0
    processed       = 0
    color_counter:  Counter = Counter()
    last_detections = None
    last_action     = "other"

    # ── Calibration pass (first 5 seconds) ────────────────────────────────
    calib_frames = min(int(fps * 5), total_frames)

    # ── Main loop ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            detections = detector.detect_frame(frame, frame_number, fps)
            detections = tracker.update(detections)

            # Accumulate jersey colors for calibration
            if frame_number < calib_frames:
                for p in detections.players:
                    color_counter[p.jersey_color] += 1
                if frame_number == calib_frames - 1:
                    team_map = detector.calibrate_teams(color_counter)
                    log.info(f"Team calibration: {team_map}")

            # Classify action
            pred = classifier.classify(tracker.ball_trajectory, detections.players, frame_number)
            segmenter.update(pred, frame_number, frame_number / fps)
            last_action     = pred.action
            last_detections = detections
            processed      += 1

        # Annotate every frame (smooth output)
        annotated = _draw_annotations(
            frame,
            last_detections or type("D", (), {"players": [], "ball": None})(),
            ACTION_LABELS.get(last_action, ""),
            frame_number, fps,
        )
        writer.write(annotated)
        frame_number += 1

        if progress_callback and frame_number % 30 == 0:
            progress_callback(min(frame_number / max(total_frames, 1), 0.99))

    # ── Finalize ────────────────────────────────────────────────────────────
    segmenter.flush(frame_number, frame_number / fps)
    cap.release()
    writer.release()

    team_colors = detector.team_colors
    team_a_color = next((c for c, t in team_colors.items() if t == "team_a"), "unknown")
    team_b_color = next((c for c, t in team_colors.items() if t == "team_b"), "unknown")

    if progress_callback:
        progress_callback(1.0)

    return {
        "duration":        round(duration, 2),
        "fps":             round(fps, 2),
        "total_frames":    total_frames,
        "processed_frames": processed,
        "events":          segmenter.events,
        "team_a_color":    team_a_color,
        "team_b_color":    team_b_color,
    }
