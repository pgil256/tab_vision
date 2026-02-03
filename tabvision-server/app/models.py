"""Data models for TabVision."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class Job:
    id: str
    status: str  # pending | processing | completed | failed
    created_at: datetime
    updated_at: datetime
    video_path: str
    capo_fret: int
    progress: float
    current_stage: str
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    # ROI coordinates (normalized 0-1)
    roi_x1: Optional[float] = None
    roi_y1: Optional[float] = None
    roi_x2: Optional[float] = None
    roi_y2: Optional[float] = None

    @classmethod
    def create(cls, video_path: str, capo_fret: int) -> "Job":
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            status="pending",
            created_at=now,
            updated_at=now,
            video_path=video_path,
            capo_fret=capo_fret,
            progress=0.0,
            current_stage="uploading",
            result_path=None,
            error_message=None,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
        }
