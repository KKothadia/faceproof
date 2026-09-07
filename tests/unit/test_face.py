import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.face.analyzer import FaceAnalyzer
from src.exceptions import FaceDetectionError

@pytest.fixture
def mock_cv2_models():
    """Mock cv2.FaceDetectorYN and cv2.FaceRecognizerSF"""
    with patch("cv2.FaceDetectorYN.create") as mock_detector_create, \
         patch("cv2.FaceRecognizerSF.create") as mock_recognizer_create, \
         patch("os.path.exists") as mock_exists:
        
        # Assume models exist for these tests
        mock_exists.return_value = True
        
        mock_detector = MagicMock()
        mock_detector_create.return_value = mock_detector
        
        mock_recognizer = MagicMock()
        mock_recognizer_create.return_value = mock_recognizer
        
        yield {
            "detector": mock_detector,
            "recognizer": mock_recognizer,
            "exists": mock_exists
        }

def test_invalid_model_path():
    """Test that FaceAnalyzer raises error if model paths are invalid."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FaceDetectionError, match="Detector model not found"):
            FaceAnalyzer(detector_path="bad_path.onnx")

def test_image_decode_failure(mock_cv2_models):
    """Test handling of None or empty images."""
    analyzer = FaceAnalyzer()
    
    with pytest.raises(FaceDetectionError, match="Invalid or empty image"):
        analyzer.analyze_image(None)
        
    with pytest.raises(FaceDetectionError, match="Invalid or empty image"):
        analyzer.analyze_image(np.array([]))

def test_no_face(mock_cv2_models):
    """Test behavior when no faces are detected."""
    # OpenCV detect returns (status, faces) where faces might be None if no face found
    mock_cv2_models["detector"].detect.return_value = (1, None)
    
    analyzer = FaceAnalyzer()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    results = analyzer.analyze_image(dummy_image)
    assert len(results) == 0

def test_one_face(mock_cv2_models):
    """Test detection and feature extraction for a single face."""
    # Mock face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, score]
    mock_face = np.array([[10, 10, 50, 50, 20, 20, 40, 20, 30, 30, 25, 45, 35, 45, 0.95]])
    mock_cv2_models["detector"].detect.return_value = (1, mock_face)
    
    # Mock feature extraction
    mock_cv2_models["recognizer"].alignCrop.return_value = np.zeros((112, 112, 3))
    mock_cv2_models["recognizer"].feature.return_value = np.array([[0.1, 0.2, 0.3]])
    
    analyzer = FaceAnalyzer()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    results = analyzer.analyze_image(dummy_image)
    
    assert len(results) == 1
    face_result = results[0]
    
    assert face_result.bbox == [10, 10, 50, 50]
    assert len(face_result.landmarks) == 5
    assert face_result.confidence == 0.95
    assert face_result.feature_vector == [0.1, 0.2, 0.3]
    assert face_result.quality_score > 0

def test_multiple_faces(mock_cv2_models):
    """Test detection for multiple faces in a single image."""
    mock_faces = np.array([
        [10, 10, 50, 50, 20, 20, 40, 20, 30, 30, 25, 45, 35, 45, 0.95],
        [60, 60, 30, 30, 65, 65, 85, 65, 75, 75, 70, 85, 80, 85, 0.85]
    ])
    mock_cv2_models["detector"].detect.return_value = (1, mock_faces)
    mock_cv2_models["recognizer"].feature.return_value = np.array([[0.5, 0.5, 0.5]])
    
    analyzer = FaceAnalyzer()
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    results = analyzer.analyze_image(dummy_image)
    assert len(results) == 2
    assert results[0].bbox == [10, 10, 50, 50]
    assert results[1].bbox == [60, 60, 30, 30]

def test_cosine_similarity(mock_cv2_models):
    """Test matching logic and threshold."""
    analyzer = FaceAnalyzer()
    
    # Mock match score
    mock_cv2_models["recognizer"].match.return_value = 0.5
    
    f1 = [0.1, 0.2]
    f2 = [0.1, 0.2]
    
    # default threshold is now 0.65 (configured)
    score = analyzer.compare_features(f1, f2)
    assert score == 0.5
    assert analyzer.is_match(score) is False
    # lower custom threshold allows match
    assert analyzer.is_match(score, threshold=0.4) is True
    # higher custom threshold still false
    assert analyzer.is_match(score, threshold=0.6) is False
