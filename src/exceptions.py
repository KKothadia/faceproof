class FaceProofError(Exception):
    """Base exception class for FaceProof application errors."""
    pass

class FaceDetectionError(FaceProofError):
    """Raised when face detection or processing fails."""
    pass

class SearchError(FaceProofError):
    """Raised when external search via SerpApi fails."""
    pass

class VerificationError(FaceProofError):
    """Raised when verification logic encounters an error."""
    pass

class EvidenceError(FaceProofError):
    """Raised when generating or verifying evidence manifests fails."""
    pass

class BlockchainError(FaceProofError):
    """Raised when blockchain interaction fails (e.g. anchoring to Base Sepolia)."""
    pass

class ConfigurationError(FaceProofError):
    """Raised when required configuration is missing or invalid."""
    pass

class MediaDownloadError(FaceProofError):
    """Raised when downloading media fails (e.g., timeout, 404, too large, invalid format)."""
    pass
