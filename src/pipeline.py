import os
import time
import logging
import uuid
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from src.schemas import (
    FaceResult, SearchCandidate, VerificationResult, 
    PipelineEvent, PipelineResult
)
from src.face.analyzer import FaceAnalyzer
from src.search.client import SerpApiClient
from src.verify.verifier import CandidateVerifier
from src.evidence.packager import EvidencePackager
from src.blockchain.client import BlockchainClient
from src.exceptions import FaceProofError, BlockchainError

logger = logging.getLogger(__name__)

class FaceProofPipeline:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.artifact_dir = os.path.join("artifacts", self.run_id)
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        self.events: List[PipelineEvent] = []
        
        # Modules
        self.face_analyzer = FaceAnalyzer()
        self.search_client = SerpApiClient()
        self.verifier = CandidateVerifier(face_analyzer=self.face_analyzer)
        self.evidence_packager = EvidencePackager(run_id=self.run_id, artifact_base_dir="artifacts")
        
        try:
            self.blockchain_client = BlockchainClient()
        except BlockchainError as e:
            logger.warning(f"Blockchain client init warning: {e}")
            self.blockchain_client = None
            
    def _emit(self, stage: str, status: str, message: str, start_time: float = None, details: Dict[str, Any] = None):
        duration_ms = int((time.time() - start_time) * 1000) if start_time else None
        event = PipelineEvent(
            event_type=stage,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details
        )
        self.events.append(event)
        logger.info(f"[{stage}] {status}: {message} ({duration_ms}ms)")
        
    def run(self, image_path: str) -> PipelineResult:
        result = PipelineResult(run_id=self.run_id, status="STARTED", events=self.events)
        
        try:
            # 1. INPUT IMAGE -> FACE DETECT -> FACE ENCODE
            start = time.time()
            if not os.path.exists(image_path):
                self._emit("IMAGE_LOAD", "FAILED", f"File not found: {image_path}", start)
                result.status = "FAILED"
                return result
                
            with open(image_path, "rb") as f:
                input_bytes = f.read()
                
            nparr = np.frombuffer(input_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                self._emit("IMAGE_LOAD", "FAILED", "Could not decode image", start)
                result.status = "FAILED"
                return result
                
            faces = self.face_analyzer.analyze_image(image)
            if not faces:
                self._emit("FACE_ANALYSIS", "FAILED", "No faces detected in input image", start)
                result.status = "NO_FACE"
                return result
                
            reference_face = sorted(faces, key=lambda f: f.quality_score, reverse=True)[0]
            result.reference_face = reference_face
            self._emit("FACE_ANALYSIS", "SUCCESS", f"Detected {len(faces)} faces, selected highest quality", start)
            
            # 4, 5, 6. SERPAPI UPLOAD -> GOOGLE LENS SEARCH -> CANDIDATE NORMALIZATION
            start = time.time()
            candidates = self.search_client.search_local_image(image, artifact_dir=self.artifact_dir)
            if not candidates:
                self._emit("SEARCH", "FAILED", "No candidates returned from search", start)
                result.status = "NO_CANDIDATES"
                return result
                
            self._emit("SEARCH", "SUCCESS", f"Found {len(candidates)} candidates", start)
            
            # 7, 8, 9. CANDIDATE MEDIA DOWNLOAD -> INDEPENDENT FACE VERIFICATION -> BEST SELECTION
            start = time.time()
            verification_results = self.verifier.verify_candidates(
                reference_face=reference_face,
                candidates=candidates,
                artifact_dir=self.artifact_dir
            )
            result.verification_results = verification_results
            
            passed_matches = [vr for vr in verification_results if vr.is_match]
            if not passed_matches:
                self._emit("VERIFICATION", "FAILED", "No candidates passed identity verification", start)
                result.status = "NO_MATCH"
                return result
                
            best_match = passed_matches[0]
            result.selected_candidate = best_match.candidate
            self._emit("VERIFICATION", "SUCCESS", f"Best match: {best_match.candidate.url}", start)
            
            # 10, 11. EVIDENCE MANIFEST -> SHA-256
            start = time.time()
            with open(best_match.artifact_path, "rb") as f:
                candidate_bytes = f.read()
                
            manifest, manifest_hash = self.evidence_packager.create_manifest(
                input_bytes=input_bytes,
                candidate_bytes=candidate_bytes,
                face_model=reference_face.model_identifier,
                discovered_source_url=best_match.candidate.url,
                discovered_source=best_match.candidate.source,
                discovered_title=best_match.candidate.metadata.get("title", ""),
                face_similarity=best_match.confidence_score,
                verification_threshold=best_match.threshold,
                result_status="VERIFIED",
                matched_face_index=best_match.matched_face_index
            )
            
            result.manifest_path = os.path.join(self.artifact_dir, "evidence.json")
            result.evidence_hash = manifest_hash
            result.media_hash = manifest.discovered_image_sha256
            self._emit("EVIDENCE", "SUCCESS", f"Generated manifest hash: {manifest_hash}", start)
            
            # 12. BLOCKCHAIN ANCHOR
            start = time.time()
            if not self.blockchain_client:
                self._emit("BLOCKCHAIN_ANCHOR", "FAILED", "Blockchain client not initialized", start)
                result.status = "FAILED"
                return result
                
            tx_hash, tx_msg = self.blockchain_client.anchor_evidence(
                evidence_hash=result.evidence_hash,
                media_hash=result.media_hash
            )
            result.transaction_hash = tx_hash
            self._emit("BLOCKCHAIN_ANCHOR", "SUCCESS", f"Anchored in tx {tx_hash}", start)
            
            # 13, 14. ON-CHAIN READ-BACK -> FINAL VERIFICATION
            start = time.time()
            is_valid, chain_msg = self.blockchain_client.verify_against_chain(
                evidence_hash=result.evidence_hash,
                expected_media_hash=result.media_hash
            )
            
            if not is_valid:
                self._emit("CHAIN_VERIFICATION", "FAILED", f"Mismatch: {chain_msg}", start)
                result.status = "TAMPER_OR_CHAIN_MISMATCH"
                return result
                
            record = self.blockchain_client.read_record(result.evidence_hash)
            result.chain_record = record
            result.final_verified = True
            
            self._emit("CHAIN_VERIFICATION", "SUCCESS", chain_msg, start)
            result.status = "SUCCESS"
            return result
            
        except FaceProofError as e:
            logger.error(f"Pipeline Domain Error: {e}")
            self._emit("ERROR", "FAILED", str(e))
            result.status = "FAILED"
            return result
        except Exception as e:
            logger.exception(f"Unexpected Pipeline Exception: {e}")
            self._emit("ERROR", "FAILED", f"Unexpected error: {str(e)}")
            result.status = "FAILED"
            return result
