from src.config import Config

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
