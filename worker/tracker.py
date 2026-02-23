"""
tracker.py
----------
Simple IoU-based tracker for ball and players across frames.
Keeps trajectory history for the ball so the action classifier
can reason about speed, direction, and height.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from detector import Detection, FrameDetections, PlayerInfo


@dataclass
class TrackedPlayer:
    track_id:   int
    player:     PlayerInfo
    lost_frames: int = 0


@dataclass
class BallTrajectory:
    """Sliding window of ball positions for action classification."""
    maxlen: int = 60   # ~2 seconds at 30fps

    positions:    deque = field(default_factory=lambda: deque(maxlen=60))  # [(cx, cy, frame)]
    timestamps:   deque = field(default_factory=lambda: deque(maxlen=60))

    def update(self, cx: float, cy: float, frame_number: int, timestamp: float):
        self.positions.append((cx, cy, frame_number))
        self.timestamps.append(timestamp)

    @property
    def speed(self) -> float:
        """Pixels/frame average speed over last 5 positions."""
        pts = list(self.positions)
        if len(pts) < 2:
            return 0.0
        speeds = []
        for i in range(max(0, len(pts)-5), len(pts)-1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            speeds.append(np.sqrt(dx**2 + dy**2))
        return float(np.mean(speeds)) if speeds else 0.0

    @property
    def direction(self) -> tuple[float, float]:
        """Normalized direction vector over last 5 frames."""
        pts = list(self.positions)
        if len(pts) < 2:
            return (0.0, 0.0)
        p1, p2 = pts[max(0, len(pts)-6)], pts[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norm   = np.sqrt(dx**2 + dy**2) + 1e-6
        return (dx / norm, dy / norm)

    @property
    def height_trend(self) -> float:
        """
        Positive = ball moving up (y decreasing in image coords).
        Negative = ball moving down.
        """
        pts = list(self.positions)
        if len(pts) < 3:
            return 0.0
        ys = [p[1] for p in pts[-6:]]
        return float(np.mean(np.diff(ys)))   # negative = rising

    @property
    def last_position(self) -> Optional[tuple[float, float]]:
        if not self.positions:
            return None
        p = self.positions[-1]
        return (p[0], p[1])


class SimpleTracker:
    """
    IoU-based tracker — matches detections across frames by overlap.
    Lightweight, CPU-friendly, no external deps.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self._next_id      = 0
        self._players:     dict[int, TrackedPlayer] = {}
        self.ball_trajectory = BallTrajectory()

    def update(self, detections: FrameDetections) -> FrameDetections:
        """
        Match new detections to existing tracks.
        Returns detections with track_id assigned to each player.
        """
        # ── Ball tracking ───────────────────────────────────────────
        if detections.ball:
            b = detections.ball
            self.ball_trajectory.update(b.cx, b.cy, detections.frame_number, detections.timestamp)

        # ── Player tracking (IoU matching) ──────────────────────────
        unmatched_dets  = list(range(len(detections.players)))
        matched_track_ids = set()

        for track_id, tracked in self._players.items():
            if not unmatched_dets:
                break
            best_iou, best_det_idx = 0.0, -1
            t_det = tracked.player.detection

            for det_idx in unmatched_dets:
                iou = self._iou(t_det, detections.players[det_idx].detection)
                if iou > best_iou:
                    best_iou, best_det_idx = iou, det_idx

            if best_iou >= self.iou_threshold:
                player = detections.players[best_det_idx]
                player.track_id = track_id
                tracked.player  = player
                tracked.lost_frames = 0
                matched_track_ids.add(track_id)
                unmatched_dets.remove(best_det_idx)

        # Increment lost counter for unmatched tracks
        for track_id in list(self._players.keys()):
            if track_id not in matched_track_ids:
                self._players[track_id].lost_frames += 1
                if self._players[track_id].lost_frames > self.max_lost:
                    del self._players[track_id]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            player = detections.players[det_idx]
            player.track_id = self._next_id
            self._players[self._next_id] = TrackedPlayer(
                track_id=self._next_id, player=player
            )
            self._next_id += 1

        # Rebuild player list with track IDs
        for track_id, tracked in self._players.items():
            if tracked.lost_frames == 0:
                pass  # already updated above

        detections.players = [
            self._players[tid].player
            for tid in self._players
            if self._players[tid].lost_frames == 0
        ]
        return detections

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
