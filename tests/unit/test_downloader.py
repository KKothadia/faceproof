import pytest
import cv2
import numpy as np
import requests
import os
from unittest.mock import patch, MagicMock
from requests import Response

from src.search.downloader import MediaDownloader
from src.schemas import SearchCandidate
from src.exceptions import MediaDownloadError
from src.config import config

@pytest.fixture
def dummy_candidate():
    return SearchCandidate(url="http://example.com/image.jpg", source="example.com")

@pytest.fixture
def valid_image_bytes():
    # Create a small valid JPEG byte array
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()

@patch("requests.get")
def test_successful_download(mock_get, dummy_candidate, valid_image_bytes, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(valid_image_bytes))}
    resp.iter_content.return_value = [valid_image_bytes]
    resp.url = "http://example.com/image.jpg"
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    artifact_dir = str(tmp_path / "artifacts")
    
    media = downloader.download_candidate_media(dummy_candidate, artifact_dir)
    assert media is not None
    assert media.source_url == dummy_candidate.url
    assert media.content_type == "image/jpeg"
    assert media.byte_size == len(valid_image_bytes)
    
    # Check if artifact was written
    assert os.path.exists(artifact_dir)
    files = os.listdir(artifact_dir)
    assert len(files) == 1
    assert files[0].endswith(".jpg")

@patch("requests.get")
def test_404_not_found(mock_get, dummy_candidate, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 404
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    # It should catch MediaDownloadError and return None
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None

@patch("requests.get")
def test_timeout(mock_get, dummy_candidate, tmp_path):
    mock_get.side_effect = requests.Timeout("Connection timed out")
    
    downloader = MediaDownloader()
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None

@patch("requests.get")
def test_invalid_content_type(mock_get, dummy_candidate, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.headers = {"Content-Type": "text/html"}
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None

@patch("requests.get")
def test_oversized_header(mock_get, dummy_candidate, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.headers = {"Content-Type": "image/jpeg", "Content-Length": str(int(config.MAX_MEDIA_SIZE_MB * 1024 * 1024 + 1))}
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None

@patch("requests.get")
def test_oversized_stream(mock_get, dummy_candidate, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.headers = {"Content-Type": "image/jpeg"}
    
    # 1MB chunk
    chunk = b"0" * (1024 * 1024)
    # Stream 10 chunks (10MB total) -> exceeds 8MB
    resp.iter_content.return_value = [chunk] * 10
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None

@patch("requests.get")
def test_corrupt_image(mock_get, dummy_candidate, tmp_path):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.headers = {"Content-Type": "image/jpeg"}
    resp.iter_content.return_value = [b"this is not a valid image byte stream"]
    mock_get.return_value = resp
    
    downloader = MediaDownloader()
    media = downloader.download_candidate_media(dummy_candidate, str(tmp_path))
    assert media is None
