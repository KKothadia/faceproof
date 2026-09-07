import pytest
import numpy as np
from unittest.mock import MagicMock

from src.search.multi_provider import MultiProviderSearchClient
from src.schemas import SearchCandidate
from src.exceptions import SearchError

@pytest.fixture
def image():
    return np.zeros((5, 5, 3), dtype=np.uint8)

def _candidate(url="https://example.com/x"):
    return SearchCandidate(url=url, source="example.com")

def test_no_providers_raises_at_construction():
    with pytest.raises(SearchError, match="at least one provider"):
        MultiProviderSearchClient([])

def test_single_provider_passthrough_success(image):
    provider = MagicMock()
    provider.search_local_image.return_value = [_candidate()]

    client = MultiProviderSearchClient([provider])
    results = client.search_local_image(image, artifact_dir="artifacts/run1")

    assert results == [_candidate()]
    provider.search_local_image.assert_called_once_with(image, artifact_dir="artifacts/run1")

def test_single_provider_passthrough_empty(image):
    provider = MagicMock()
    provider.search_local_image.return_value = []

    client = MultiProviderSearchClient([provider])
    assert client.search_local_image(image) == []

def test_falls_through_to_second_provider_when_first_returns_empty(image):
    primary = MagicMock()
    primary.search_local_image.return_value = []
    secondary = MagicMock()
    secondary.search_local_image.return_value = [_candidate("https://fallback.example.com/y")]

    client = MultiProviderSearchClient([primary, secondary])
    results = client.search_local_image(image)

    assert len(results) == 1
    assert results[0].url == "https://fallback.example.com/y"
    primary.search_local_image.assert_called_once()
    secondary.search_local_image.assert_called_once()

def test_falls_through_to_second_provider_when_first_errors(image):
    primary = MagicMock()
    primary.search_local_image.side_effect = SearchError("SerpApi is down")
    secondary = MagicMock()
    secondary.search_local_image.return_value = [_candidate()]

    client = MultiProviderSearchClient([primary, secondary])
    results = client.search_local_image(image)

    assert len(results) == 1
    secondary.search_local_image.assert_called_once()

def test_primary_result_wins_secondary_never_called(image):
    """A real result from the first provider must never be overridden or padded by a second."""
    primary = MagicMock()
    primary.search_local_image.return_value = [_candidate("https://primary.example.com")]
    secondary = MagicMock()

    client = MultiProviderSearchClient([primary, secondary])
    results = client.search_local_image(image)

    assert results[0].url == "https://primary.example.com"
    secondary.search_local_image.assert_not_called()

def test_all_providers_return_empty_yields_empty_list(image):
    primary = MagicMock()
    primary.search_local_image.return_value = []
    secondary = MagicMock()
    secondary.search_local_image.return_value = []

    client = MultiProviderSearchClient([primary, secondary])
    assert client.search_local_image(image) == []

def test_all_providers_error_raises_last_error(image):
    primary = MagicMock()
    primary.search_local_image.side_effect = SearchError("primary down")
    secondary = MagicMock()
    secondary.search_local_image.side_effect = SearchError("secondary down too")

    client = MultiProviderSearchClient([primary, secondary])
    with pytest.raises(SearchError, match="secondary down too"):
        client.search_local_image(image)

def test_error_then_clean_empty_result_returns_empty_not_error(image):
    """If the first provider errors but the second completes cleanly with zero results,
    that's a genuine 'search ran, nothing found' outcome - not an error to surface."""
    primary = MagicMock()
    primary.search_local_image.side_effect = SearchError("primary down")
    secondary = MagicMock()
    secondary.search_local_image.return_value = []

    client = MultiProviderSearchClient([primary, secondary])
    assert client.search_local_image(image) == []


def test_duplicate_provider_results_combined(image):
    """Both providers can return candidates; multi_provider returns first provider's results
    when they are non-empty (no cross-provider dedup needed at this layer)."""
    primary = MagicMock()
    primary.search_local_image.return_value = [
        _candidate("https://example.com/a"),
        _candidate("https://example.com/b"),
    ]
    secondary = MagicMock()

    client = MultiProviderSearchClient([primary, secondary])
    results = client.search_local_image(image)

    # Primary wins; secondary never called
    assert len(results) == 2
    secondary.search_local_image.assert_not_called()


def test_provider_failure_fallback_to_secondary(image):
    """SearchError from primary triggers fallback to secondary."""
    primary = MagicMock()
    primary.search_local_image.side_effect = SearchError("rate limited")
    secondary = MagicMock()
    secondary.search_local_image.return_value = [_candidate("https://fallback.com")]

    client = MultiProviderSearchClient([primary, secondary])
    results = client.search_local_image(image)

    assert len(results) == 1
    assert results[0].url == "https://fallback.com"
