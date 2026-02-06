
from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    video_id: str
    excluded_timeframes: List[list]