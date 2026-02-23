"""
action_classifier.py
--------------------
Rule-based action classifier using ball trajectory and player positions.

Actions classified:
  - SERVE   : ball starts near back line, high speed, directed to opponent court
  - ATTACK  : fast downward ball near net after apex
  - BLOCK   : ball bounced sharply upward near net, players' hands high
  - SET     : slow upward ball between two teammates
  - PASS    : moderate speed, upward, away from net
  - DIG     : very low ball position, upward rebound, low player posture
  - OTHER   : anything that doesn't match above

The classifier operates on a sliding window of ball trajectory data
and the current player positions, running every frame.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from tracker import BallTrajectory
from detector import PlayerInfo


class ActionType(str, Enum):
    SERVE  = "serve"
    PASS   = "pass"
    SET    = "set"
    ATTACK = "attack"
    BLOCK  = "block"
    DIG    = "dig"
    OTHER  = "other"


@dataclass
class ActionPrediction:
    action:     ActionType
    confidence: float        # 0.0 → 1.0
    description: str = ""


# Tunable thresholds (normalized coordinates 0-1 for positions)
THRESHOLDS = {
    "net_x":           0.48,   # normalized x of net (center of frame)
    "back_line_x":     0.15,   # near back line
    "high_y":          0.35,   # ball considered "high" when cy < 35% height
    "low_y":           0.75,   # ball considered "low" when cy > 75% height
    "fast_speed":      12.0,   # pixels/frame
    "slow_speed":      4.0,
    "very_fast_speed": 20.0,
}


class ActionClassifier:
    """
    Stateful classifier — call update() every frame, returns current action.
    Maintains a short event history to detect action transitions.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.fw = frame_width
        self.fh = frame_height
        self._last_action  = ActionType.OTHER
        self._stable_count = 0   # frames the current action has been stable
        self._event_buffer: list[ActionPrediction] = []

    def classify(
        self,
        trajectory: BallTrajectory,
        players:    list[PlayerInfo],
        frame_number: int,
    ) -> ActionPrediction:
        """
        Classify the current action based on ball trajectory + player context.
        Returns ActionPrediction with action type and confidence.
        """
        if not trajectory.positions:
            return ActionPrediction(ActionType.OTHER, 0.0, "No ball detected")

        ball_pos = trajectory.last_position
        if ball_pos is None:
            return ActionPrediction(ActionType.OTHER, 0.0, "No ball position")

        # Normalize ball position
        bx = ball_pos[0] / self.fw
        by = ball_pos[1] / self.fh

        speed        = trajectory.speed
        h_trend      = trajectory.height_trend   # negative = rising
        dx, dy       = trajectory.direction

        # ── Player context ───────────────────────────────────────────
        n_players     = len(players)
        players_near_net = sum(
            1 for p in players
            if abs(p.detection.cx / self.fw - THRESHOLDS["net_x"]) < 0.15
        )
        players_arms_up = sum(
            1 for p in players
            if p.detection.y1 / self.fh < 0.4   # very high detection top
        )

        # ── Classification rules ─────────────────────────────────────

        # SERVE: ball starts near back line, high speed, horizontal trajectory
        if (bx < THRESHOLDS["back_line_x"] or bx > 1 - THRESHOLDS["back_line_x"]) \
                and speed > THRESHOLDS["fast_speed"] \
                and abs(dx) > 0.7:
            return ActionPrediction(
                ActionType.SERVE, 0.82,
                f"Ball near back line ({bx:.2f}), speed={speed:.1f}px/f, horizontal"
            )

        # ATTACK: fast downward ball near net
        if speed > THRESHOLDS["fast_speed"] \
                and dy > 0.4 \
                and abs(bx - THRESHOLDS["net_x"]) < 0.25 \
                and by < THRESHOLDS["high_y"] + 0.2:
            return ActionPrediction(
                ActionType.ATTACK, 0.85,
                f"Fast downward ball near net, speed={speed:.1f}, dy={dy:.2f}"
            )

        # BLOCK: ball suddenly going upward near net, arms raised
        if h_trend < -3 \
                and abs(bx - THRESHOLDS["net_x"]) < 0.2 \
                and players_arms_up >= 1:
            return ActionPrediction(
                ActionType.BLOCK, 0.78,
                f"Ball rebounding upward near net, h_trend={h_trend:.2f}"
            )

        # SET: slow ball moving upward, mid-court
        if speed < THRESHOLDS["slow_speed"] \
                and h_trend < -1.5 \
                and THRESHOLDS["back_line_x"] < bx < 1 - THRESHOLDS["back_line_x"]:
            return ActionPrediction(
                ActionType.SET, 0.74,
                f"Slow upward arc, speed={speed:.1f}, h_trend={h_trend:.2f}"
            )

        # DIG: ball very low, moving upward from near floor
        if by > THRESHOLDS["low_y"] and h_trend < -2:
            return ActionPrediction(
                ActionType.DIG, 0.76,
                f"Ball low ({by:.2f}), rising from floor area"
            )

        # PASS: moderate speed, upward, no special context
        if THRESHOLDS["slow_speed"] < speed < THRESHOLDS["fast_speed"] \
                and h_trend < 0:
            return ActionPrediction(
                ActionType.PASS, 0.65,
                f"Moderate upward ball, speed={speed:.1f}"
            )

        return ActionPrediction(
            ActionType.OTHER, 0.5,
            f"No clear pattern. speed={speed:.1f}, h_trend={h_trend:.2f}"
        )


class EventSegmenter:
    """
    Groups frame-level predictions into discrete events with start/end times.
    An event is confirmed after MIN_STABLE_FRAMES consistent predictions.
    """

    MIN_STABLE_FRAMES = 8
    MIN_EVENT_DURATION = 0.3   # seconds

    def __init__(self):
        self._buffer:    list[tuple[ActionType, float, int, float]] = []
        self.events:     list[dict] = []
        self._current_action = ActionType.OTHER
        self._start_frame    = 0
        self._start_time     = 0.0
        self._stable_count   = 0

    def update(self, prediction: ActionPrediction, frame_number: int, timestamp: float):
        action = prediction.action

        if action == self._current_action:
            self._stable_count += 1
        else:
            # Potential transition — commit current event if long enough
            if self._stable_count >= self.MIN_STABLE_FRAMES:
                duration = timestamp - self._start_time
                if duration >= self.MIN_EVENT_DURATION and self._current_action != ActionType.OTHER:
                    self.events.append({
                        "action_type":   self._current_action,
                        "start_frame":   self._start_frame,
                        "end_frame":     frame_number,
                        "start_time":    self._start_time,
                        "end_time":      timestamp,
                        "confidence":    prediction.confidence,
                        "description":   prediction.description,
                    })
            self._current_action = action
            self._start_frame    = frame_number
            self._start_time     = timestamp
            self._stable_count   = 1

    def flush(self, final_frame: int, final_time: float):
        """Call at end of video to flush the last event."""
        if self._stable_count >= self.MIN_STABLE_FRAMES \
                and self._current_action != ActionType.OTHER:
            self.events.append({
                "action_type":  self._current_action,
                "start_frame":  self._start_frame,
                "end_frame":    final_frame,
                "start_time":   self._start_time,
                "end_time":     final_time,
                "confidence":   0.6,
                "description":  "Last event (flush)",
            })
