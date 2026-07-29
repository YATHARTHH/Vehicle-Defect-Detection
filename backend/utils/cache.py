import hashlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("overbody_api.cache")

# Local in-memory cache dictionary: sha256 -> response_dict
_memory_cache: Dict[str, Dict[str, Any]] = {}


def compute_image_hash(image_bytes: bytes) -> str:
    """Computes SHA-256 hash of image bytes for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


def get_cached_response(image_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached response dict if present."""
    if image_hash in _memory_cache:
        logger.info(f"[cache-hit] In-memory cache hit for hash {image_hash[:10]}...")
        cached = dict(_memory_cache[image_hash])
        cached["cache_hit"] = True
        return cached
    return None


def set_cached_response(image_hash: str, response_dict: Dict[str, Any]) -> None:
    """Stores response dict in cache."""
    _memory_cache[image_hash] = response_dict
    logger.info(f"[cache-store] Stored analysis in cache for hash {image_hash[:10]}...")
