"""In-memory job storage."""
from typing import Dict, Optional

from app.models import Job


class JobStorage:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def save(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)


# Global instance for the application
job_storage = JobStorage()
