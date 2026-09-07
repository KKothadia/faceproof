import os
import json
import logging
from typing import Any, Dict, Optional

from src.config import config

logger = logging.getLogger(__name__)

CACHE_ROOT = os.path.join(".cache", "search")

def _cache_path(provider: str, image_hash: str) -> str:
    return os.path.join(CACHE_ROOT, provider, f"{image_hash}.json")

def read(provider: str, image_hash: str) -> Optional[Dict[str, Any]]:
    """Namespaced by provider so two providers' incompatible response shapes never collide."""
    if not config.SEARCH_CACHE_ENABLED:
        return None
    path = _cache_path(provider, image_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable search cache entry %s: %s", path, e)
        return None

def write(provider: str, image_hash: str, data: Dict[str, Any]) -> None:
    if not config.SEARCH_CACHE_ENABLED:
        return
    try:
        path = _cache_path(provider, image_hash)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning("Could not write search cache: %s", e)
