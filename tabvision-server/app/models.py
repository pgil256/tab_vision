"""Data models for TabVision."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional
import uuid

Instrument = Literal["acoustic", "electric", "classical"]
Tone = Literal["clean", "distorted"]
PlayingStyle = Literal["fingerstyle", "strumming", "mixed"]
AccuracyMode = Literal["fast", "accurate"]


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
    instrument: Instrument = "acoustic"
    tone: Tone = "clean"
    style: PlayingStyle = "mixed"
    accuracy_mode: AccuracyMode = "accurate"
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    # ROI coordinates (normalized 0-1)
    roi_x1: Optional[float] = None
    roi_y1: Optional[float] = None
    roi_x2: Optional[float] = None
    roi_y2: Optional[float] = None
    # Whether the processing pipeline runs the video stack for this job.
    # None until processing starts (the pipeline config decides, not the
    # upload); lets the client hide video stages for audio-only runs.
    video_enabled: Optional[bool] = None

    @classmethod
    def create(
        cls,
        video_path: str,
        capo_fret: int,
        *,
        instrument: Instrument = "acoustic",
        tone: Tone = "clean",
        style: PlayingStyle = "mixed",
        accuracy_mode: AccuracyMode = "accurate",
    ) -> "Job":
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
            instrument=instrument,
            tone=tone,
            style=style,
            accuracy_mode=accuracy_mode,
            result_path=None,
            error_message=None,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "instrument": self.instrument,
            "tone": self.tone,
            "style": self.style,
            "accuracy_mode": self.accuracy_mode,
            "error_message": self.error_message,
            "video_enabled": self.video_enabled,
        }

    def to_record(self) -> dict:
        """Serialize the full job for durable storage."""
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "video_path": self.video_path,
            "capo_fret": self.capo_fret,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "instrument": self.instrument,
            "tone": self.tone,
            "style": self.style,
            "accuracy_mode": self.accuracy_mode,
            "result_path": self.result_path,
            "error_message": self.error_message,
            "roi_x1": self.roi_x1,
            "roi_y1": self.roi_y1,
            "roi_x2": self.roi_x2,
            "roi_y2": self.roi_y2,
            "video_enabled": self.video_enabled,
        }

    @classmethod
    def from_record(cls, record: dict) -> "Job":
        """Rehydrate a job from durable storage."""
        return cls(
            id=record["id"],
            status=record["status"],
            created_at=datetime.fromisoformat(record["created_at"]),
            updated_at=datetime.fromisoformat(record["updated_at"]),
            video_path=record["video_path"],
            capo_fret=record["capo_fret"],
            progress=record["progress"],
            current_stage=record["current_stage"],
            instrument=record.get("instrument", "acoustic"),
            tone=record.get("tone", "clean"),
            style=record.get("style", "mixed"),
            accuracy_mode=record.get("accuracy_mode", "accurate"),
            result_path=record.get("result_path"),
            error_message=record.get("error_message"),
            roi_x1=record.get("roi_x1"),
            roi_y1=record.get("roi_y1"),
            roi_x2=record.get("roi_x2"),
            roi_y2=record.get("roi_y2"),
            video_enabled=record.get("video_enabled"),
        )
