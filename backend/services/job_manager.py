import uuid
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("overbody_api.job_manager")

# In-memory Async Job Store: job_id -> job_dict
_jobs_db: Dict[str, Dict[str, Any]] = {}


def create_job(filename: str) -> str:
    """Creates a new async inspection job record in PENDING status."""
    job_id = str(uuid.uuid4())
    _jobs_db[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "status": "PENDING",  # PENDING | PROCESSING | COMPLETED | FAILED
        "progress": 0,
        "result": None,
        "error": None,
    }
    logger.info(f"[job-manager] Created async job {job_id[:8]} for {filename}")
    return job_id


def update_job(
    job_id: str,
    status: str,
    progress: int = 100,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Updates job status, progress percentage, and result payload."""
    if job_id in _jobs_db:
        _jobs_db[job_id]["status"] = status
        _jobs_db[job_id]["progress"] = progress
        if result:
            _jobs_db[job_id]["result"] = result
        if error:
            _jobs_db[job_id]["error"] = error
        logger.info(f"[job-manager] Updated job {job_id[:8]} status -> {status} ({progress}%)")


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves async job status record."""
    return _jobs_db.get(job_id)
