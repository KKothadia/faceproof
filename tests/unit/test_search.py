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
        
def test_large_image_compression():
    """Test logic for compressing an oversized image."""
    # Create large image
    large_image = np.zeros((3000, 3000, 3), dtype=np.uint8)
    
    client = SerpApiClient(api_key="test_key")
    compressed = client._compress_image(large_image)
    
    assert len(compressed) <= 500 * 1024
