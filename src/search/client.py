import os
import json
import logging
import cv2
import numpy as np
import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from src.config import config
from src.schemas import SearchCandidate
from src.exceptions import SearchError

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB limit for SerpApi Image API

class SerpApiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.SERPAPI_API_KEY
        if not self.api_key or self.api_key == "your_serpapi_key_here":
            raise SearchError("SerpApi API key is not configured or is set to default.")
        self.upload_url = "https://serpapi.com/image" 
        self.search_url = "https://serpapi.com/search.json"
        self.timeout = 15

    def _compress_image(self, image: np.ndarray) -> bytes:
        """Compress/re-encode to JPEG under 500 KB limit."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise SearchError("Invalid image provided for compression.")
            
        quality = 95
        while quality > 10:
            success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not success:
                raise SearchError("Failed to encode image to JPEG.")
                
            image_bytes = encoded_image.tobytes()
            if len(image_bytes) <= MAX_FILE_SIZE_BYTES:
                return image_bytes
            quality -= 5
            
        # If still over limit, resize
        scale = 0.8
        while len(image_bytes) > MAX_FILE_SIZE_BYTES and scale > 0.1:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
            success, encoded_image = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if success:
                image_bytes = encoded_image.tobytes()
            scale -= 0.1
            
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise SearchError("Could not compress image under 500KB.")
            
        return image_bytes

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle HTTP status codes and return JSON payload."""
        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError:
                raise SearchError("Invalid JSON response from API.")
        elif status in (401, 403):
            logger.error("Authentication failed or forbidden. Status: %d", status)
            raise SearchError(f"Authentication error: {status}")
        elif status == 429:
            logger.error("Rate limit exceeded.")
            raise SearchError("Rate limit exceeded (429).")
        elif 500 <= status < 600:
            logger.error("Server error: %d", status)
            raise SearchError(f"SerpApi server error: {status}")
        else:
            raise SearchError(f"Unexpected API error: {status} - {response.text}")

    def _upload_image(self, image_bytes: bytes) -> str:
        """Upload image to get image_id, retrying on transient failures."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info("Uploading image to SerpApi (attempt %d/%d)", attempt + 1, max_retries)
                response = requests.post(
                    self.upload_url,
                    files={"image": ("search.jpg", image_bytes, "image/jpeg")},
                    data={"api_key": self.api_key},
                    timeout=self.timeout
                )
                
                # Retry only on 5xx transient server errors
                if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
                data = self._handle_response(response)
                image_id = data.get("image_id")
                if not image_id:
                    raise SearchError("No image_id returned from upload endpoint.")
                return image_id
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error("Request exception during upload: %s", str(e))
                raise SearchError(f"Network error during upload: {e}")
                
        raise SearchError("Failed to upload image after retries.")

    def _search_google_lens(self, image_id: str) -> Dict[str, Any]:
        """Query Google Lens engine with the image_id."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info("Querying Google Lens with image_id (attempt %d/%d)", attempt + 1, max_retries)
                params = {
                    "engine": "google_lens",
                    "image_id": image_id,
                    "api_key": self.api_key
                }
                response = requests.get(
                    self.search_url,
                    params=params,
                    timeout=self.timeout
                )
                
                if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
                return self._handle_response(response)
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error("Request exception during search: %s", str(e))
                raise SearchError(f"Network error during search: {e}")
                
        raise SearchError("Failed to search image after retries.")

    def _save_artifact(self, data: Dict[str, Any], artifact_dir: str):
        """Save raw response with secrets redacted for debugging."""
        try:
            os.makedirs(artifact_dir, exist_ok=True)
            safe_data = data.copy()
            # Redact any accidental keys in search parameters
            if "search_parameters" in safe_data and isinstance(safe_data["search_parameters"], dict):
                safe_data["search_parameters"] = {
                    k: ("REDACTED" if k == "api_key" else v)
                    for k, v in safe_data["search_parameters"].items()
                }
            
            filepath = os.path.join(artifact_dir, f"serpapi_response_{int(time.time())}.json")
            with open(filepath, "w") as f:
                json.dump(safe_data, f, indent=2)
            logger.info("Saved raw API response to %s", filepath)
        except Exception as e:
            logger.warning("Failed to save artifact: %s", str(e))

    def _parse_candidates(self, raw_response: Dict[str, Any]) -> List[SearchCandidate]:
        """Extract and normalize candidate matches, deduplicating by URL."""
        candidates: Dict[str, SearchCandidate] = {}
        
        def add_result(item: Dict[str, Any], source_type: str):
            url = item.get("link")
            if not url:
                return
                
            if url in candidates:
                return
                
            domain = urlparse(url).netloc
            metadata = {
                "title": item.get("title", ""),
                "domain": domain,
                "position": len(candidates) + 1,
                "match_type": source_type
            }
            
            candidates[url] = SearchCandidate(
                url=url,
                source=domain,
                thumbnail_url=item.get("thumbnail"),
                metadata=metadata
            )

        # Parse visual matches
        for item in raw_response.get("visual_matches", []):
            add_result(item, "visual_match")
            
        # Parse exact matches
        for item in raw_response.get("exact_matches", []):
            add_result(item, "exact_match")
            
        # Parse organic results
        for item in raw_response.get("organic_results", []):
            add_result(item, "organic_result")
            
        return list(candidates.values())

    def search_local_image(self, image: np.ndarray, artifact_dir: Optional[str] = None) -> List[SearchCandidate]:
        """Perform the complete reverse-image-search flow."""
        compressed_bytes = self._compress_image(image)
        image_id = self._upload_image(compressed_bytes)
        raw_response = self._search_google_lens(image_id)
        
        if artifact_dir:
            self._save_artifact(raw_response, artifact_dir)
            
        return self._parse_candidates(raw_response)
