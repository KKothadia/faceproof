import pytest

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
