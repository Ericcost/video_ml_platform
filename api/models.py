"""
models.py — Pydantic schemas for the Volleyball Analyzer API
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING   = "pending"
    PROCESSING = "processing"
    DONE      = "done"
    ERROR     = "error"


class ActionType(str, Enum):
    SERVE    = "serve"
    PASS     = "pass"
    SET      = "set"
    ATTACK   = "attack"
    BLOCK    = "block"
    DIG      = "dig"
    OTHER    = "other"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PlayerDetection(BaseModel):
    player_id:  int
    team:       str          # "team_a" | "team_b" | "unknown"
    jersey_number: Optional[str] = None
    bbox:       BoundingBox
    confidence: float


class BallDetection(BaseModel):
    bbox:       BoundingBox
    confidence: float
    position:   list[float]  # [cx, cy] normalized


class ActionEvent(BaseModel):
    action_type:  ActionType
    start_frame:  int
    end_frame:    int
    start_time:   float      # seconds
    end_time:     float
    confidence:   float
    players_involved: list[int] = []
    team:         Optional[str] = None
    description:  str = ""


class FrameResult(BaseModel):
    frame_number: int
    timestamp:    float
    players:      list[PlayerDetection] = []
    ball:         Optional[BallDetection] = None
    action:       Optional[ActionType] = None


class AnalysisResult(BaseModel):
    job_id:          str
    video_filename:  str
    duration:        float
    fps:             float
    total_frames:    int
    processed_frames: int
    events:          list[ActionEvent] = []
    team_a_color:    str = "unknown"
    team_b_color:    str = "unknown"
    output_video_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id:   str
    status:   JobStatus
    message:  str = ""
    progress: float = 0.0     # 0.0 → 1.0
    result:   Optional[AnalysisResult] = None
