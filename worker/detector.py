"""
detector.py
-----------
YOLOv8-based detector for:
  - Volleyball (class 'sports ball' in COCO, id=32)
  - Players (class 'person', id=0)

Also classifies team by jersey color using HSV clustering.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional

COCO_PERSON = 0
COCO_BALL   = 32   # sports ball

# HSV ranges for jersey color detection.
# Red wraps around in HSV so it needs two ranges merged via bitwise OR.
# Thresholds are intentionally a bit loose to handle gym/indoor lighting.
COLOR_PROFILES = {
    "red":    ([0,   80,  80],  [10,  255, 255]),
    "red2":   ([165, 80,  80],  [180, 255, 255]),   # red upper wrap-around
    "blue":   ([95,  80,  40],  [135, 255, 255]),
    "white":  ([0,   0,   170], [180, 55,  255]),
    "black":  ([0,   0,   0],   [180, 255, 60]),
    "yellow": ([18,  80,  80],  [38,  255, 255]),
    "green":  ([35,  60,  60],  [85,  255, 255]),
    "orange": ([8,   80,  80],  [20,  255, 255]),
    "purple": ([125, 40,  40],  [165, 255, 255]),
}


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    class_id: int

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def w(self):  return self.x2 - self.x1
    @property
    def h(self):  return self.y2 - self.y1
    @property
    def area(self): return self.w * self.h


@dataclass
class PlayerInfo:
    detection: Detection
    team: str = "unknown"
    jersey_color: str = "unknown"
    jersey_number: Optional[str] = None
    track_id: int = -1


@dataclass
class FrameDetections:
    frame_number: int
    timestamp: float
    players: list[PlayerInfo] = field(default_factory=list)
    ball: Optional[Detection] = None


class VolleyballDetector:
    def __init__(self, model_path: str = "../models/yolov8n.pt", device: str = "cpu"):
        self.model  = YOLO(model_path)
        self.device = device
        # Team color calibration (set after first N frames)
        self.team_colors: dict[str, str] = {}   # color_name → "team_a" | "team_b"
        self._color_samples: list[str]   = []

    def detect_frame(self, frame: np.ndarray, frame_number: int = 0, fps: float = 25.0) -> FrameDetections:
        """Run YOLOv8 inference on a single frame."""
        results = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=0.3,
            iou=0.45,
        )[0]

        players: list[PlayerInfo] = []
        ball:    Optional[Detection] = None

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            det = Detection(x1, y1, x2, y2, conf, cls)

            if cls == COCO_BALL and (ball is None or conf > ball.confidence):
                ball = det

            elif cls == COCO_PERSON:
                # Extract jersey color from torso region
                color = self._get_jersey_color(frame, det)
                team  = self.team_colors.get(color, "unknown")
                players.append(PlayerInfo(
                    detection=det,
                    jersey_color=color,
                    team=team,
                ))

        return FrameDetections(
            frame_number=frame_number,
            timestamp=frame_number / max(fps, 1),
            players=players,
            ball=ball,
        )

    def _get_jersey_color(self, frame: np.ndarray, det: Detection) -> str:
        """
        Extract dominant jersey color from the upper 40% of the player bbox.
        Uses HSV masking against predefined color profiles.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(det.x1))
        y1 = max(0, int(det.y1))
        x2 = min(w, int(det.x2))
        # Only use torso area (top 40%)
        y2 = min(h, int(det.y1 + det.h * 0.4))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi  = frame[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        best_color, best_score = "unknown", 0

        # Merge red2 into red (HSV wrap-around)
        color_scores: dict[str, float] = {}
        for color_name, (lower, upper) in COLOR_PROFILES.items():
            mask  = cv2.inRange(hsv, np.array(lower), np.array(upper))
            score = float(np.sum(mask)) / (mask.size * 255 + 1e-6)   # normalise 0→1
            base  = "red" if color_name == "red2" else color_name
            color_scores[base] = color_scores.get(base, 0.0) + score

        for color_name, score in color_scores.items():
            if score > best_score:
                best_score = score
                best_color = color_name

        return best_color if best_score > 0.05 else "unknown"

    def calibrate_teams(self, color_counts: dict[str, int]) -> dict[str, str]:
        """
        After sampling N frames, auto-assign the two most frequent
        jersey colors to team_a and team_b.
        """
        # Sort by frequency, take top 2
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        top2 = [c for c, _ in sorted_colors if c != "unknown"][:2]

        if len(top2) >= 2:
            self.team_colors = {top2[0]: "team_a", top2[1]: "team_b"}
        elif len(top2) == 1:
            self.team_colors = {top2[0]: "team_a"}

        return self.team_colors
