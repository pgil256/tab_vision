"""Modal-backed job storage."""
from typing import MutableMapping, Optional

from app.models import Job


class ModalJobStorage:
    """Persist jobs in a Modal Dict-compatible mapping."""

    def __init__(self, records: MutableMapping):
        self._records = records

    def save(self, job: Job) -> None:
        self._records[job.id] = job.to_record()

    def get(self, job_id: str) -> Optional[Job]:
        record = self._records.get(job_id)
        if record is None:
            return None
        return Job.from_record(record)
