import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

from src.config import config
from src.schemas import FaceResult
from src.exceptions import FaceDetectionError

class FaceAnalyzer:
    """Wrapper for OpenCV YuNet detector and SFace recognizer."""
    
    def __init__(self, detector_path: str = None, recognizer_path: str = None, detection_threshold: float = None):
        self.detector_path = detector_path or config.MODEL_YUNET_PATH
        self.recognizer_path = recognizer_path or config.MODEL_SFACE_PATH
        self.detection_threshold = detection_threshold if detection_threshold is not None else config.FACE_DETECTION_THRESHOLD
        
        if not os.path.exists(self.detector_path):
            raise FaceDetectionError(f"Detector model not found at {self.detector_path}")
        if not os.path.exists(self.recognizer_path):
            raise FaceDetectionError(f"Recognizer model not found at {self.recognizer_path}")
            
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.detector_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.detection_threshold,
                nms_threshold=0.3,
                top_k=5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.recognizer_path,
                config=""
            )
        except Exception as e:
            raise FaceDetectionError(f"Failed to initialize face models: {e}")

    def analyze_image(self, image: np.ndarray) -> List[FaceResult]:
        """Detect faces and extract features from an image."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise FaceDetectionError("Invalid or empty image provided")
            
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        
        try:
            status, faces = self.detector.detect(image)
        except Exception as e:
            raise FaceDetectionError(f"Detection failed: {e}")
            
        results = []
        if faces is None:
            return results
            
        for face in faces:
            bbox = face[0:4].astype(int).tolist()
            
            # YuNet returns 5 landmarks: right eye, left eye, nose, right mouth, left mouth
            landmarks = []
            for i in range(5):
                landmarks.append([int(face[4 + i*2]), int(face[4 + i*2 + 1])])
                
            confidence = float(face[14])
            
            # Align and extract features
            try:
                aligned_face = self.recognizer.alignCrop(image, face)
                feature = self.recognizer.feature(aligned_face)
                feature_vector = feature[0].tolist()
            except Exception as e:
                raise FaceDetectionError(f"Feature extraction failed: {e}")
                
            # Quality calculation heuristic: confidence scaled by face size relative to image size
            face_area = bbox[2] * bbox[3]
            img_area = width * height
            quality_score = confidence * (face_area / img_area if img_area > 0 else 0)
            
            results.append(FaceResult(
                bbox=bbox,
                landmarks=landmarks,
                feature_vector=feature_vector,
                confidence=confidence,
                quality_score=quality_score,
                model_identifier="yunet_sface"
            ))
            
        return results

    def compare_features(self, feature1: List[float], feature2: List[float]) -> float:
        """Calculate cosine similarity between two feature vectors."""
        if not feature1 or not feature2:
            raise FaceDetectionError("Invalid features provided for comparison")
            
        f1 = np.array(feature1, dtype=np.float32).reshape(1, -1)
        f2 = np.array(feature2, dtype=np.float32).reshape(1, -1)
        
        try:
            score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)
        except Exception as e:
            raise FaceDetectionError(f"Feature comparison failed: {e}")
            
    def is_match(self, score: float, threshold: float = None) -> bool:
        """Determine if two faces match based on cosine similarity score."""
        t = threshold if threshold is not None else config.FACE_MATCH_COSINE_THRESHOLD
        return score >= t
