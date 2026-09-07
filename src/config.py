import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    BASE_SEPOLIA_RPC_URL: str = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
    
    # Face Analysis Configuration
    # Note: Thresholds are empirical/model-specific and not universal.
    FACE_DETECTION_THRESHOLD: float = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.80"))
    FACE_MATCH_COSINE_THRESHOLD: float = float(os.getenv("FACE_MATCH_COSINE_THRESHOLD", "0.65"))
    # OpenCV 4.x: use face_detection_yunet_2023mar.onnx (fixed-input, works with cv2 DNN legacy engine)
    # OpenCV 5.x: the 2023mar model silently returns no detections with the new dnn5 engine.
    #             Install opencv-contrib-python==4.10.0.84 instead (recommended), or set
    #             MODEL_YUNET_PATH=models/face_detection_yunet_2026may.onnx in .env
    MODEL_YUNET_PATH: str = os.getenv("MODEL_YUNET_PATH", "models/face_detection_yunet_2023mar.onnx")
    MODEL_SFACE_PATH: str = os.getenv("MODEL_SFACE_PATH", "models/face_recognition_sface_2021dec.onnx")
    
    # Media Downloader Configuration
    MAX_MEDIA_SIZE_MB: float = float(os.getenv("MAX_MEDIA_SIZE_MB", "8.0"))
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration variables are set."""
        missing = []
        if not cls.SERPAPI_API_KEY:
            missing.append("SERPAPI_API_KEY")
        if not cls.PRIVATE_KEY:
            missing.append("PRIVATE_KEY")
            
        if missing:
            logger.warning(f"Warning: Missing configuration for {', '.join(missing)}")

config = Config()
