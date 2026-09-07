import os
import hashlib
import logging
import cv2
import numpy as np
import requests
from typing import Optional

from src.config import config
from src.schemas import SearchCandidate, DownloadedMedia
from src.exceptions import MediaDownloadError

logger = logging.getLogger(__name__)

class MediaDownloader:
    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.max_size_bytes = int(config.MAX_MEDIA_SIZE_MB * 1024 * 1024)
        # Normal User-Agent to prevent basic blocks
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

    def download_candidate_media(self, candidate: SearchCandidate, artifact_dir: str) -> Optional[DownloadedMedia]:
        """
        Download, validate, and save media from a candidate URL safely.
        Returns None if download fails, allowing pipeline to skip without aborting.
        """
        # Prefer thumbnail if available (more likely to be direct image), fallback to url
        target_url = candidate.thumbnail_url or candidate.url
        
        try:
            return self._download(target_url, artifact_dir)
        except MediaDownloadError as e:
            logger.warning("Skipping candidate %s due to download failure: %s", target_url, str(e))
            return None
            
    def _download(self, url: str, artifact_dir: str) -> DownloadedMedia:
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                stream=True, 
                allow_redirects=True, 
                timeout=self.timeout
            )
        except requests.Timeout:
            raise MediaDownloadError("Download timed out")
        except requests.RequestException as e:
            raise MediaDownloadError(f"Request failed: {e}")
            
        status = response.status_code
        if status == 404:
            raise MediaDownloadError("Media not found (404)")
        elif status == 403:
            raise MediaDownloadError("Access forbidden (403)")
        elif status == 429:
            raise MediaDownloadError("Rate limit exceeded (429)")
        elif status != 200:
            raise MediaDownloadError(f"HTTP Error {status}")
            
        content_type = response.headers.get("Content-Type", "")
        if not content_type.lower().startswith("image/"):
            raise MediaDownloadError(f"HTML or invalid content type: {content_type}")
            
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_size_bytes:
            raise MediaDownloadError(f"File too large in header: {content_length} bytes")
            
        chunks = []
        downloaded_size = 0
        hasher = hashlib.sha256()
        
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                downloaded_size += len(chunk)
                hasher.update(chunk)
                
                if downloaded_size > self.max_size_bytes:
                    raise MediaDownloadError("Oversized file detected during stream")
        except requests.RequestException as e:
            raise MediaDownloadError(f"Stream read failed: {e}")
            
        content_bytes = b"".join(chunks)
        if downloaded_size == 0:
            raise MediaDownloadError("Empty file downloaded")
            
        # Validate by decoding safely with OpenCV
        nparr = np.frombuffer(content_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise MediaDownloadError("Corrupt image or unsupported format")
            
        sha256_hash = hasher.hexdigest()
        
        # Save safely to artifact dir (not executing, just writing standard jpeg)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, f"{sha256_hash}.jpg")
        
        try:
            # Write out the cleanly decoded image matrix, shedding arbitrary payload bytes
            cv2.imwrite(artifact_path, img)
        except Exception as e:
            raise MediaDownloadError(f"Failed to save artifact locally: {e}")
            
        return DownloadedMedia(
            source_url=url,
            final_url=response.url,
            content_bytes=content_bytes,
            sha256_hash=sha256_hash,
            content_type=content_type,
            byte_size=downloaded_size
        )
