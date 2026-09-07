import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def _get_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number, got: {raw!r}")

def _get_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got: {raw!r}")

class Config:
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    BASE_SEPOLIA_RPC_URL: str = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")

    # Face Analysis Configuration
    # Note: Thresholds are empirical/model-specific and not universal.
    FACE_DETECTION_THRESHOLD: float = _get_float("FACE_DETECTION_THRESHOLD", "0.80")
    FACE_MATCH_COSINE_THRESHOLD: float = _get_float("FACE_MATCH_COSINE_THRESHOLD", "0.65")
    # OpenCV 4.x: use face_detection_yunet_2023mar.onnx (fixed-input, works with cv2 DNN legacy engine)
    # OpenCV 5.x: the 2023mar model silently returns no detections with the new dnn5 engine.
    #             Install opencv-contrib-python==4.10.0.84 instead (recommended), or set
    #             MODEL_YUNET_PATH=models/face_detection_yunet_2026may.onnx in .env
    MODEL_YUNET_PATH: str = os.getenv("MODEL_YUNET_PATH", "models/face_detection_yunet_2023mar.onnx")
    MODEL_SFACE_PATH: str = os.getenv("MODEL_SFACE_PATH", "models/face_recognition_sface_2021dec.onnx")

    # Media Downloader Configuration
    MAX_MEDIA_SIZE_MB: float = _get_float("MAX_MEDIA_SIZE_MB", "8.0")

    # Search Configuration
    # Caps how many normalized candidates the pipeline will download and face-verify per run,
    # so a large Google Lens result set can't silently balloon runtime/bandwidth.
    MAX_SEARCH_RESULTS: int = _get_int("MAX_SEARCH_RESULTS", "20")
    # Cache identical repeated searches (same input image) on disk instead of re-hitting
    # SerpApi every time - saves API quota during iterative demo/testing runs.
    SEARCH_CACHE_ENABLED: bool = os.getenv("SEARCH_CACHE_ENABLED", "true").strip().lower() not in ("false", "0", "")

    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration. Missing API/keys are only warned about (many unit tests and
        offline dev flows run without them). Out-of-range numeric values are a real
        misconfiguration and raise, since they'd otherwise silently corrupt pipeline behavior.
        """
        missing = []
        if not cls.SERPAPI_API_KEY:
            missing.append("SERPAPI_API_KEY")
        if not cls.PRIVATE_KEY:
            missing.append("PRIVATE_KEY")

        if missing:
            logger.warning(f"Warning: Missing configuration for {', '.join(missing)}")

        if not (0.0 < cls.FACE_DETECTION_THRESHOLD <= 1.0):
            raise ConfigurationError(
                f"FACE_DETECTION_THRESHOLD must be in (0, 1], got {cls.FACE_DETECTION_THRESHOLD}"
            )
        if not (0.0 < cls.FACE_MATCH_COSINE_THRESHOLD <= 1.0):
            raise ConfigurationError(
                f"FACE_MATCH_COSINE_THRESHOLD must be in (0, 1], got {cls.FACE_MATCH_COSINE_THRESHOLD}"
            )
        if cls.MAX_MEDIA_SIZE_MB <= 0:
            raise ConfigurationError(f"MAX_MEDIA_SIZE_MB must be > 0, got {cls.MAX_MEDIA_SIZE_MB}")
        if cls.MAX_SEARCH_RESULTS <= 0:
            raise ConfigurationError(f"MAX_SEARCH_RESULTS must be > 0, got {cls.MAX_SEARCH_RESULTS}")

        # The RPC URL carries the signed transaction; require HTTPS for any non-local endpoint
        # so it can't be silently downgraded to plaintext (localhost is exempted for local/
        # simulated chains such as Anvil/Hardhat, per the project's "any blockchain" allowance).
        parsed_rpc = urlparse(cls.BASE_SEPOLIA_RPC_URL)
        is_local = parsed_rpc.hostname in ("localhost", "127.0.0.1", "::1")
        if not is_local and parsed_rpc.scheme != "https":
            raise ConfigurationError(
                f"BASE_SEPOLIA_RPC_URL must use https:// (got scheme {parsed_rpc.scheme!r}): "
                f"{cls.BASE_SEPOLIA_RPC_URL}"
            )

config = Config()
