import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from src.schemas import EvidenceManifest

def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()

def canonicalize_json(data: Dict[str, Any]) -> str:
    """Canonicalize JSON strictly as per requirements."""
    return json.dumps(
        data, 
        sort_keys=True, 
        separators=(",", ":"), 
        ensure_ascii=False
    )

class EvidencePackager:
    def __init__(self, run_id: str, artifact_base_dir: str = "artifacts"):
        self.run_id = run_id
        self.run_dir = os.path.join(artifact_base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
    def create_manifest(self, 
                        input_bytes: bytes, 
                        candidate_bytes: bytes, 
                        face_model: str, 
                        discovered_source_url: str,
                        discovered_source: str,
                        discovered_title: str,
                        face_similarity: float,
                        verification_threshold: float,
                        result_status: str,
                        matched_face_index: int = None,
                        search_request_id: str = None) -> Tuple[EvidenceManifest, str]:
        
        # Calculate hashes of raw inputs
        input_sha256 = compute_sha256(input_bytes)
        candidate_sha256 = compute_sha256(candidate_bytes)
        
        manifest = EvidenceManifest(
            schema_version="1.0",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            pipeline_version="1.0.0",
            face_model=face_model,
            input_sha256=input_sha256,
            discovered_source_url=discovered_source_url,
            discovered_source=discovered_source,
            discovered_title=discovered_title,
            discovered_image_sha256=candidate_sha256,
            search_provider="SerpApi Google Lens",
            search_request_id=search_request_id,
            face_similarity=face_similarity,
            verification_threshold=verification_threshold,
            matched_face_index=matched_face_index,
            result_status=result_status
        )
        
        # Save candidate hash separately as evidence
        with open(os.path.join(self.run_dir, "candidate.sha256"), "w") as f:
            f.write(candidate_sha256)
            
        # Serialize to dict to canonicalize
        manifest_dict = manifest.model_dump()
        canonical_json = canonicalize_json(manifest_dict)
        
        # Hash canonical JSON with SHA-256
        manifest_hash = compute_sha256(canonical_json.encode("utf-8"))
        
        # Save canonical JSON
        with open(os.path.join(self.run_dir, "evidence.json"), "w", encoding="utf-8") as f:
            f.write(canonical_json)
            
        # Save JSON hash
        with open(os.path.join(self.run_dir, "evidence.sha256"), "w", encoding="utf-8") as f:
            f.write(manifest_hash)
            
        return manifest, manifest_hash

def verify_manifest(path: str, expected_hash: str) -> bool:
    """Verify that the evidence.json file matches the expected SHA-256 hash."""
    if not os.path.exists(path):
        return False
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    computed_hash = compute_sha256(content.encode("utf-8"))
    return computed_hash == expected_hash
