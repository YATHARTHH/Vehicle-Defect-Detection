import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api_access.jsonl")

logger = logging.getLogger("overbody_api.json_logger")


def log_api_request(
    request_id: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    defects_found: int = 0,
    image_hash: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    """Logs structured JSON lines to logs/api_access.jsonl."""
    log_entry: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "defects_found": defects_found,
        "image_hash": image_hash,
        "error_code": error_code,
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write structured JSON log: {e}")
