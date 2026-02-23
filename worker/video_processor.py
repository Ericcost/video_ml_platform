"""
video_processor.py  — v2
Fix: re-encode en H264 via ffmpeg après le processing OpenCV
Fix: events enrichis avec action_type en string pour Plotly
"""

import cv2
import os
import shutil
import subprocess
import logging
import numpy as np
from collections import Counter
from typing import Callable, Optional

from detector import VolleyballDetector
from tracker import SimpleTracker
from action_classifier import ActionClassifier, EventSegmenter

log = logging.getLogger(__name__)

COLORS = {
    "team_a":  (220, 80,  30),
    "team_b":  (30,  100, 220),
    "unknown": (160, 160, 160),
    "ball":    (0,   240, 240),
    "action":  (255, 255,  50),
}

ACTION_LABELS = {
    "serve":  "SERVE",
    "pass":   "PASS",
    "set":    "SET",
    "attack": "ATTACK",
    "block":  "BLOCK",
    "dig":    "DIG",
    "other":  "",
}


class _EmptyDetections:
    players = []
    ball    = None


def _draw_annotations(frame, detections, action_label, frame_number, fps):
    out = frame.copy()
    h, w = out.shape[:2]

    for player in detections.players:
        d     = player.detection
        color = COLORS.get(player.team, COLORS["unknown"])
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{player.track_id}"
        if player.team != "unknown":
            label += f" ({player.team[-1].upper()})"
        lw = len(label) * 8
        cv2.rectangle(out, (x1, y1 - 20), (x1 + lw, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if detections.ball:
        b  = detections.ball
        cx, cy = int(b.cx), int(b.cy)
        r  = max(8, int(min(b.w, b.h) / 2))
        cv2.circle(out, (cx, cy), r, COLORS["ball"], 2)
        cv2.circle(out, (cx, cy), 3, COLORS["ball"], -1)

    if action_label:
        banner_w = max(200, len(action_label) * 22 + 20)
        cv2.rectangle(out, (0, h - 60), (banner_w, h), (0, 0, 0), -1)
        cv2.putText(out, action_label, (12, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, COLORS["action"], 2, cv2.LINE_AA)

    ts     = frame_number / max(fps, 1)
    ts_str = f"{int(ts // 60):02d}:{ts % 60:05.2f}"
    cv2.putText(out, ts_str, (w - 135, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def _reencode_h264(src: str, dst: str) -> bool:
    """Re-encode mp4v → H264/yuv420p pour compatibilité navigateur et Streamlit."""
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0:
            log.info("H264 re-encode OK")
            return True
        log.warning(f"ffmpeg stderr: {result.stderr.decode()[:500]}")
        return False
    except Exception as e:
        log.warning(f"ffmpeg error: {e}")
        return False


def process_video(
    input_path:        str,
    output_path:       str,
    progress_callback: Optional[Callable[[float], None]] = None,
    frame_skip:        int = 2,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / max(fps, 1)

    log.info(f"Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames — {duration:.1f}s")

    detector   = VolleyballDetector(model_path="yolov8n.pt", device="cpu")
    tracker    = SimpleTracker()
    classifier = ActionClassifier(frame_width=width, frame_height=height)
    segmenter  = EventSegmenter()

    raw_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    frame_number    = 0
    processed       = 0
    color_counter: Counter = Counter()
    last_detections = _EmptyDetections()
    last_action     = "other"
    calibrated      = False
    calib_frames    = min(int(fps * 5), total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            detections = detector.detect_frame(frame, frame_number, fps)
            detections = tracker.update(detections)

            if frame_number < calib_frames:
                # Accumulate jersey color samples during calibration window
                for p in detections.players:
                    if p.jersey_color != "unknown":
                        color_counter[p.jersey_color] += 1
            elif not calibrated:
                # Trigger exactly once, on the first processed frame after the window
                calibrated = True
                team_map = detector.calibrate_teams(color_counter)
                log.info(f"Team calibration: {team_map} (from {sum(color_counter.values())} samples)")

            pred = classifier.classify(
                tracker.ball_trajectory, detections.players, frame_number
            )
            segmenter.update(pred, frame_number, frame_number / fps)
            last_action     = pred.action.value   # store string value, not enum
            last_detections = detections
            processed      += 1

        annotated = _draw_annotations(
            frame, last_detections,
            ACTION_LABELS.get(last_action, ""),
            frame_number, fps,
        )
        writer.write(annotated)
        frame_number += 1

        if progress_callback and frame_number % 30 == 0:
            progress_callback(min(frame_number / max(total_frames, 1), 0.92))

    segmenter.flush(frame_number, frame_number / fps)
    cap.release()
    writer.release()

    if progress_callback:
        progress_callback(0.94)

    # Re-encode to H264
    encoded_ok = _reencode_h264(raw_path, output_path)
    if encoded_ok:
        try:
            os.remove(raw_path)
        except Exception:
            pass
    else:
        shutil.move(raw_path, output_path)
        log.warning("Fallback: using raw mp4v (H264 re-encode failed)")

    if progress_callback:
        progress_callback(1.0)

    team_colors  = detector.team_colors
    team_a_color = next((c for c, t in team_colors.items() if t == "team_a"), "unknown")
    team_b_color = next((c for c, t in team_colors.items() if t == "team_b"), "unknown")

    events = [
        {**ev, "action_type": str(ev["action_type"]), "players_involved": [], "team": None}
        for ev in segmenter.events
    ]

    return {
        "duration":         round(duration, 2),
        "fps":              round(fps, 2),
        "total_frames":     total_frames,
        "processed_frames": processed,
        "events":           events,
        "team_a_color":     team_a_color,
        "team_b_color":     team_b_color,
    }
