"""Fake data generation for testing the skeleton."""
import uuid
from datetime import datetime, timezone


def generate_fake_tab_document(job_id: str, capo_fret: int) -> dict:
    """Generate a fake TabDocument for testing the skeleton."""
    return {
        "id": job_id,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration": 30.0,
        "capoFret": capo_fret,
        "tuning": ["E", "A", "D", "G", "B", "E"],
        "notes": [
            {
                "id": str(uuid.uuid4()),
                "timestamp": 0.5,
                "string": 1,
                "fret": 0,
                "confidence": 0.95,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 1.0,
                "string": 2,
                "fret": 1,
                "confidence": 0.72,
                "confidenceLevel": "medium",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 1.5,
                "string": 3,
                "fret": 0,
                "confidence": 0.88,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 2.0,
                "string": 4,
                "fret": 2,
                "confidence": 0.45,
                "confidenceLevel": "low",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 2.5,
                "string": 5,
                "fret": 3,
                "confidence": 0.91,
                "confidenceLevel": "high",
                "isEdited": False,
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": 3.0,
                "string": 6,
                "fret": 3,
                "confidence": 0.67,
                "confidenceLevel": "medium",
                "isEdited": False,
            },
        ],
    }
