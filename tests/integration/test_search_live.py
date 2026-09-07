import pytest
import numpy as np
import os
from src.search.client import SerpApiClient
from src.exceptions import SearchError
from src.config import config

@pytest.mark.live
def test_live_serpapi_search():
    """
    Live search using actual SerpApi endpoint.
    Run with: python -m pytest tests/integration/test_search_live.py --live-search
    """
    if not config.SERPAPI_API_KEY or config.SERPAPI_API_KEY == "your_serpapi_key_here":
        pytest.skip("Valid SERPAPI_API_KEY not configured")
        
    # Generate a noise image that should upload cleanly
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    client = SerpApiClient()
    try:
        # Saving artifacts to see raw response during manual verification
        results = client.search_local_image(image, artifact_dir="artifacts/")
        assert isinstance(results, list)
    except SearchError as e:
        pytest.fail(f"Live search failed with SearchError: {e}")
