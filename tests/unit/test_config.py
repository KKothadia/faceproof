import pytest
from unittest.mock import patch

from src.config import Config
from src.exceptions import ConfigurationError

def test_config_initialization():
    """Test that the configuration class can be initialized without errors."""
    config = Config()
    assert config is not None
    assert hasattr(config, "BASE_SEPOLIA_RPC_URL")

def test_config_defaults():
    """Test the default values in configuration."""
    config = Config()
    assert config.BASE_SEPOLIA_RPC_URL == "https://sepolia.base.org"

def test_config_validate_no_exceptions():
    """Test that validate doesn't raise exceptions on default state."""
    config = Config()
    # It prints warnings but shouldn't crash
    config.validate()

@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, 5.0])
def test_config_validate_rejects_out_of_range_detection_threshold(threshold):
    with patch.object(Config, "FACE_DETECTION_THRESHOLD", threshold):
        with pytest.raises(ConfigurationError, match="FACE_DETECTION_THRESHOLD"):
            Config.validate()

@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, 5.0])
def test_config_validate_rejects_out_of_range_match_threshold(threshold):
    with patch.object(Config, "FACE_MATCH_COSINE_THRESHOLD", threshold):
        with pytest.raises(ConfigurationError, match="FACE_MATCH_COSINE_THRESHOLD"):
            Config.validate()

def test_config_validate_rejects_non_positive_max_media_size():
    with patch.object(Config, "MAX_MEDIA_SIZE_MB", 0):
        with pytest.raises(ConfigurationError, match="MAX_MEDIA_SIZE_MB"):
            Config.validate()

def test_config_validate_rejects_non_positive_max_search_results():
    with patch.object(Config, "MAX_SEARCH_RESULTS", 0):
        with pytest.raises(ConfigurationError, match="MAX_SEARCH_RESULTS"):
            Config.validate()

def test_config_default_max_search_results():
    assert Config.MAX_SEARCH_RESULTS == 20

def test_config_validate_rejects_non_https_rpc_url():
    with patch.object(Config, "BASE_SEPOLIA_RPC_URL", "http://sepolia.base.org"):
        with pytest.raises(ConfigurationError, match="BASE_SEPOLIA_RPC_URL must use https"):
            Config.validate()

def test_config_validate_allows_http_localhost_rpc_url():
    """A local/simulated chain (e.g. Anvil/Hardhat) is exempt from the HTTPS requirement."""
    with patch.object(Config, "BASE_SEPOLIA_RPC_URL", "http://127.0.0.1:8545"):
        Config.validate()

def test_config_validate_allows_default_https_rpc_url():
    Config.validate()
