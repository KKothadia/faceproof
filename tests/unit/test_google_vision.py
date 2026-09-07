import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from requests import Response

from src.search.vision_client import GoogleVisionClient
from src.exceptions import SearchError

@pytest.fixture
def mock_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)

def test_missing_api_key_raises():
    with pytest.raises(SearchError, match="not configured"):
        GoogleVisionClient(api_key="")

def test_default_placeholder_key_raises():
    with pytest.raises(SearchError, match="not configured"):
        GoogleVisionClient(api_key="your_google_vision_api_key_here")

@patch("requests.post")
def test_search_success_parses_web_detection(mock_post, mock_image):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.json.return_value = {
        "responses": [
            {
                "webDetection": {
                    "pagesWithMatchingImages": [
                        {"url": "https://social.example.com/post/1", "pageTitle": "A matching post"},
                        {"url": "https://social.example.com/post/1", "pageTitle": "duplicate"},
                    ],
                    "fullMatchingImages": [
                        {"url": "https://cdn.example.com/exact.jpg"}
                    ],
                }
            }
        ]
    }
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    results = client.search_local_image(mock_image)

    assert len(results) == 2  # deduped
    assert results[0].url == "https://social.example.com/post/1"
    assert results[0].metadata["match_type"] == "page_with_matching_image"
    assert results[0].metadata["provider"] == "google_cloud_vision"
    assert results[1].url == "https://cdn.example.com/exact.jpg"

@patch("requests.post")
def test_empty_web_detection_returns_no_candidates(mock_post, mock_image):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.json.return_value = {"responses": [{}]}
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    assert client.search_local_image(mock_image) == []

@patch("requests.post")
def test_malformed_response_returns_no_candidates(mock_post, mock_image):
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.json.return_value = {}  # no "responses" key at all
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    assert client.search_local_image(mock_image) == []

@patch("requests.post")
def test_auth_error(mock_post, mock_image):
    resp = MagicMock(spec=Response)
    resp.status_code = 403
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    with pytest.raises(SearchError, match="authentication error: 403"):
        client.search_local_image(mock_image)

@patch("requests.post")
def test_rate_limit(mock_post, mock_image):
    resp = MagicMock(spec=Response)
    resp.status_code = 429
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    with pytest.raises(SearchError, match="rate limit exceeded"):
        client.search_local_image(mock_image)

@patch("requests.post")
def test_api_level_error_in_response_body(mock_post, mock_image):
    """Google Vision returns HTTP 200 with an embedded error object for some failures."""
    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.json.return_value = {"responses": [{"error": {"message": "Invalid image content"}}]}
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    with pytest.raises(SearchError, match="Invalid image content"):
        client.search_local_image(mock_image)

@patch("requests.post")
def test_results_capped_at_max_search_results(mock_post, mock_image):
    from src.config import config as app_config

    resp = MagicMock(spec=Response)
    resp.status_code = 200
    resp.json.return_value = {
        "responses": [
            {
                "webDetection": {
                    "pagesWithMatchingImages": [
                        {"url": f"https://example.com/page{i}", "pageTitle": f"Page {i}"}
                        for i in range(app_config.MAX_SEARCH_RESULTS + 10)
                    ]
                }
            }
        ]
    }
    mock_post.return_value = resp

    client = GoogleVisionClient(api_key="test_key")
    results = client.search_local_image(mock_image)

    assert len(results) == app_config.MAX_SEARCH_RESULTS
