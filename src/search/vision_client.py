import os
import json
import time
import base64
import hashlib
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from src.config import config
from src.schemas import SearchCandidate
from src.exceptions import SearchError
from src.search import cache as search_cache

logger = logging.getLogger(__name__)

PROVIDER_NAME = "google_cloud_vision"
ANNOTATE_URL = "https://vision.googleapis.com/v1/images:annotate"

class GoogleVisionClient:
    """
    Secondary reverse-image-search provider using Google Cloud Vision's Web Detection
    feature (free tier: 1,000 lookups/month). Unlike SerpApi's upload-then-poll flow,
    Vision's REST API takes the image as inline base64 bytes directly in the request body -
    no public image hosting required, keeping the "local-only" constraint intact.

    Used by src.search.multi_provider.MultiProviderSearchClient as a fallback when the
    primary provider (SerpApi/Google Lens) returns zero candidates or fails outright.
    """

    def __init__(self, api_key: Optional[str] = None):
        # If caller explicitly passes a key (even empty string), use it directly;
        # only fall back to env config when the parameter is None.
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = config.GOOGLE_VISION_API_KEY
        if not self.api_key or self.api_key == "your_google_vision_api_key_here":
            raise SearchError("Google Cloud Vision API key is not configured.")
        self.timeout = 15

    def _encode_image(self, image: np.ndarray) -> str:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise SearchError("Invalid image provided for Google Vision.")
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise SearchError("Failed to encode image for Google Vision.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        status = response.status_code
        if status in (401, 403):
            logger.error("Google Vision authentication failed or forbidden. Status: %d", status)
            raise SearchError(f"Google Vision authentication error: {status}")
        if status == 429:
            logger.error("Google Vision rate limit exceeded.")
            raise SearchError("Google Vision rate limit exceeded (429).")
        if 500 <= status < 600:
            logger.error("Google Vision server error: %d", status)
            raise SearchError(f"Google Vision server error: {status}")
        if status != 200:
            raise SearchError(f"Unexpected Google Vision API error: {status} - {response.text}")

        try:
            data = response.json()
        except ValueError:
            raise SearchError("Invalid JSON response from Google Vision.")

        responses = data.get("responses", [])
        if responses and "error" in responses[0]:
            err = responses[0]["error"]
            raise SearchError(f"Google Vision API error: {err.get('message', err)}")

        return data

    def _call_api(self, image_b64: str) -> Dict[str, Any]:
        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "WEB_DETECTION", "maxResults": config.MAX_SEARCH_RESULTS},
                                 {"type": "SAFE_SEARCH_DETECTION"}],
                }
            ]
        }
        try:
            response = requests.post(
                ANNOTATE_URL,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.error("Request exception during Google Vision call: %s", str(e))
            raise SearchError(f"Network error calling Google Vision: {e}")

        return self._handle_response(response)

    def _save_artifact(self, data: Dict[str, Any], artifact_dir: str) -> None:
        """Save raw response for debugging. The API key lives only in the request URL, not
        anywhere in this response body, so no redaction is needed here."""
        try:
            os.makedirs(artifact_dir, exist_ok=True)
            filepath = os.path.join(artifact_dir, f"google_vision_response_{int(time.time())}.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved raw Google Vision response to %s", filepath)
        except OSError as e:
            logger.warning("Failed to save Google Vision artifact: %s", str(e))

    def _parse_candidates(self, data: Dict[str, Any]) -> List[SearchCandidate]:
        """Normalize Web Detection's pages/matching-images into SearchCandidate, deduping by URL."""
        candidates: Dict[str, SearchCandidate] = {}
        responses = data.get("responses", [])
        web_detection = responses[0].get("webDetection", {}) if responses else {}

        def add(url: str, title: str, match_type: str):
            if not url or url in candidates:
                return
            domain = urlparse(url).netloc
            candidates[url] = SearchCandidate(
                url=url,
                source=domain,
                thumbnail_url=None,
                metadata={
                    "title": title,
                    "domain": domain,
                    "position": len(candidates) + 1,
                    "match_type": match_type,
                    "provider": PROVIDER_NAME,
                },
            )

        # Pages that embed a matching/near-matching copy of this exact image - the closest
        # analogue to SerpApi's "exact_matches".
        for page in web_detection.get("pagesWithMatchingImages", []):
            add(page.get("url", ""), page.get("pageTitle", ""), "page_with_matching_image")

        # Direct image URLs Vision considers full/partial matches, with no hosting page context.
        for img in web_detection.get("fullMatchingImages", []):
            add(img.get("url", ""), "", "full_matching_image")
        for img in web_detection.get("partialMatchingImages", []):
            add(img.get("url", ""), "", "partial_matching_image")

        results = list(candidates.values())
        if len(results) > config.MAX_SEARCH_RESULTS:
            results = results[:config.MAX_SEARCH_RESULTS]
        return results

    def search_local_image(self, image: np.ndarray, artifact_dir: Optional[str] = None) -> List[SearchCandidate]:
        """Same content-hash cache as SerpApiClient, namespaced separately so the two
        providers' differently-shaped raw responses never collide."""
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()

        data = search_cache.read(PROVIDER_NAME, image_hash)
        if data is not None:
            logger.info("Search cache hit for image hash %s - skipping live Google Vision call", image_hash)
        else:
            image_b64 = self._encode_image(image)
            data = self._call_api(image_b64)
            search_cache.write(PROVIDER_NAME, image_hash, data)

        if artifact_dir:
            self._save_artifact(data, artifact_dir)

        return self._parse_candidates(data)
