import os
import cv2
import numpy as np
import logging
from typing import List, Optional

from src.schemas import FaceResult, SearchCandidate, VerificationResult
from src.face.analyzer import FaceAnalyzer
from src.search.downloader import MediaDownloader
from src.config import config
from src.exceptions import VerificationError

logger = logging.getLogger(__name__)

class CandidateVerifier:
    def __init__(self, face_analyzer: Optional[FaceAnalyzer] = None, downloader: Optional[MediaDownloader] = None):
        self.face_analyzer = face_analyzer or FaceAnalyzer()
        self.downloader = downloader or MediaDownloader()
        self.threshold = config.FACE_MATCH_COSINE_THRESHOLD

    def verify_candidates(self, reference_face: FaceResult, candidates: List[SearchCandidate], artifact_dir: str) -> List[VerificationResult]:
        """
        Verify search candidates against a reference face purely based on SFace similarity.
        Silently skips invalid candidates and keeps processing.
        Returns deterministically sorted VerificationResults.
        """
        if not reference_face.feature_vector:
            raise VerificationError("Reference face must have a computed feature vector.")
            
        results = []
        for candidate in candidates:
            # 1. Download media
            media = self.downloader.download_candidate_media(candidate, artifact_dir)
            if not media:
                logger.info("Skipping candidate %s: failed to download media.", candidate.url)
                continue
                
            # Reconstruct image from bytes safely
            nparr = np.frombuffer(media.content_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.info("Skipping candidate %s: corrupt image.", candidate.url)
                continue
                
            artifact_path = os.path.join(artifact_dir, f"{media.sha256_hash}.jpg")
                
            # 2. Detect and process faces
            try:
                candidate_faces = self.face_analyzer.analyze_image(image)
            except Exception as e:
                logger.warning("Skipping candidate %s: face analysis error - %s", candidate.url, str(e))
                continue
                
            num_faces = len(candidate_faces)
            best_face = None
            best_score = 0.0
            best_idx = None
            
            # 3, 4, 5, 6. Compute similarity and select local best face
            for idx, c_face in enumerate(candidate_faces):
                if c_face.feature_vector:
                    score = self.face_analyzer.compare_features(
                        reference_face.feature_vector, 
                        c_face.feature_vector
                    )
                    if score > best_score:
                        best_score = score
                        best_face = c_face
                        best_idx = idx
                        
            # 7, 8. Pass/Fail threshold identity check
            is_match = best_score >= self.threshold
            if num_faces == 0:
                reason = "No faces found in candidate image"
            elif is_match:
                reason = "Face similarity meets or exceeds threshold"
            else:
                reason = "Face similarity below threshold"
            
            # Compute deterministic ranking score
            # Primary: Face similarity
            combined_rank = float(best_score)
            
            # Bonus: Exact-match source mapping
            if candidate.metadata.get("match_type") == "exact_match":
                combined_rank += 0.05
                
            # Tie-breaker: original Search position (lower position = subtract less)
            position = candidate.metadata.get("position", 100)
            combined_rank -= (position * 0.0001)
            
            result = VerificationResult(
                is_match=is_match,
                confidence_score=best_score,
                candidate=candidate,
                face_details=best_face,
                matched_face_index=best_idx,
                number_of_faces=num_faces,
                threshold=self.threshold,
                pass_reason=reason,
                artifact_path=artifact_path,
                combined_rank_score=combined_rank
            )
            results.append(result)
            
        # Sort by combined deterministic rank score descending
        results.sort(key=lambda r: r.combined_rank_score, reverse=True)
        return results
