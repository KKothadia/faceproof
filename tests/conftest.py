import pytest

from src.config import config as app_config

@pytest.fixture(autouse=True)
def _disable_search_cache(monkeypatch):
    """
    Several tests reuse identical synthetic images (same content hash) with different
    mocked SerpApi responses. The on-disk search cache (src/search/client.py) is
    content-addressed by image hash, so leaving it enabled would let one test's cached
    response leak into another and make outcomes depend on execution order. Tests must
    always exercise the live (mocked) request path deterministically.
    """
    monkeypatch.setattr(app_config, "SEARCH_CACHE_ENABLED", False)

def pytest_addoption(parser):
    parser.addoption(
        "--live-search", action="store_true", default=False, help="run tests that make live API calls"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "live: mark test as requiring a live API")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--live-search"):
        return
    skip_live = pytest.mark.skip(reason="need --live-search option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
