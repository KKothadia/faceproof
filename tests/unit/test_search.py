import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from requests import Response

from src.search.client import SerpApiClient
from src.exceptions import SearchError

@pytest.fixture
def mock_image():
    # 10x10 dummy image
    return np.zeros((10, 10, 3), dtype=np.uint8)

@patch("requests.post")
@patch("requests.get")
def test_search_success_mock(mock_get, mock_post, mock_image):
    """Test full flow with mocked responses."""
    # Mock upload
    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 200
    post_resp.json.return_value = {"image_id": "dummy_123"}
    mock_post.return_value = post_resp
    
    # Mock search
    get_resp = MagicMock(spec=Response)
    get_resp.status_code = 200
    get_resp.json.return_value = {
        "visual_matches": [
            {"link": "https://example.com/match1", "title": "Match 1", "thumbnail": "thumb1"},
            {"link": "https://example.com/match1", "title": "Duplicate"} # deduplication check
        ],
        "exact_matches": [
            {"link": "https://example.com/exact", "title": "Exact 1"}
        ]
    }
    mock_get.return_value = get_resp
    
    client = SerpApiClient(api_key="test_key")
    results = client.search_local_image(mock_image)
    
    assert len(results) == 2
    
    # visual matches get parsed first
    assert results[0].url == "https://example.com/match1"
    assert results[0].metadata["domain"] == "example.com"
    assert results[0].metadata["match_type"] == "visual_match"
    
    # exact matches get parsed next
    assert results[1].url == "https://example.com/exact"
    assert results[1].metadata["match_type"] == "exact_match"

@patch("requests.post")
def test_upload_retry_and_fail(mock_post, mock_image):
    """Test retries on 5xx errors."""
    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 500
    mock_post.return_value = post_resp
    
    client = SerpApiClient(api_key="test_key")
    with pytest.raises(SearchError, match="SerpApi server error: 500"):
        client.search_local_image(mock_image)
        
    assert mock_post.call_count == 3 # 3 retries max

@patch("requests.post")
def test_auth_error(mock_post, mock_image):
    """Test explicit 403 handling (no retry)."""
    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 403
    mock_post.return_value = post_resp
    
    client = SerpApiClient(api_key="test_key")
    with pytest.raises(SearchError, match="Authentication error: 403"):
        client.search_local_image(mock_image)
    
    # Should only try once for 403
    assert mock_post.call_count == 1
    
@patch("requests.post")
def test_rate_limit(mock_post, mock_image):
    """Test explicit 429 handling."""
    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 429
    mock_post.return_value = post_resp
    
    client = SerpApiClient(api_key="test_key")
    with pytest.raises(SearchError, match="Rate limit exceeded"):
        client.search_local_image(mock_image)
        
@patch("requests.post")
@patch("requests.get")
def test_malformed_response_returns_empty_list(mock_get, mock_post, mock_image):
    """A response missing all expected keys must not crash - just yield zero candidates."""
    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 200
    post_resp.json.return_value = {"image_id": "dummy_123"}
    mock_post.return_value = post_resp

    get_resp = MagicMock(spec=Response)
    get_resp.status_code = 200
    get_resp.json.return_value = {"search_metadata": {"status": "Success"}}  # no match keys at all
    mock_get.return_value = get_resp

    client = SerpApiClient(api_key="test_key")
    results = client.search_local_image(mock_image)

    assert results == []

@patch("requests.post")
@patch("requests.get")
def test_candidates_capped_at_max_search_results(mock_get, mock_post, mock_image):
    """A large result set must be capped so downstream download/verification stays bounded."""
    from src.config import config as app_config

    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 200
    post_resp.json.return_value = {"image_id": "dummy_123"}
    mock_post.return_value = post_resp

    many_matches = [
        {"link": f"https://example.com/match{i}", "title": f"Match {i}"}
        for i in range(app_config.MAX_SEARCH_RESULTS + 15)
    ]
    get_resp = MagicMock(spec=Response)
    get_resp.status_code = 200
    get_resp.json.return_value = {"visual_matches": many_matches}
    mock_get.return_value = get_resp

    client = SerpApiClient(api_key="test_key")
    results = client.search_local_image(mock_image)

    assert len(results) == app_config.MAX_SEARCH_RESULTS

@patch("requests.post")
@patch("requests.get")
def test_identical_image_search_is_served_from_cache(mock_get, mock_post, mock_image, tmp_path, monkeypatch):
    """A second search for the same image content must not hit the network again."""
    monkeypatch.setattr("src.search.client.SEARCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("src.config.config.SEARCH_CACHE_ENABLED", True)

    post_resp = MagicMock(spec=Response)
    post_resp.status_code = 200
    post_resp.json.return_value = {"image_id": "dummy_123"}
    mock_post.return_value = post_resp

    get_resp = MagicMock(spec=Response)
    get_resp.status_code = 200
    get_resp.json.return_value = {
        "visual_matches": [{"link": "https://example.com/cached", "title": "Cached match"}]
    }
    mock_get.return_value = get_resp

    client = SerpApiClient(api_key="test_key")

    first = client.search_local_image(mock_image)
    assert mock_post.call_count == 1
    assert mock_get.call_count == 1
    assert first[0].url == "https://example.com/cached"

    second = client.search_local_image(mock_image)
    # No additional network calls for the identical image
    assert mock_post.call_count == 1
    assert mock_get.call_count == 1
    assert second[0].url == "https://example.com/cached"

def test_large_image_compression():
    """Test logic for compressing an oversized image."""
    # Create large image
    large_image = np.zeros((3000, 3000, 3), dtype=np.uint8)
    
    client = SerpApiClient(api_key="test_key")
    compressed = client._compress_image(large_image)
    
    assert len(compressed) <= 500 * 1024
