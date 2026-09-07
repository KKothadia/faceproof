from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class FaceResult(BaseModel):
    bbox: List[int]
    landmarks: List[List[int]]
    feature_vector: Optional[List[float]] = None
    confidence: float
    quality_score: float = 0.0
    model_identifier: str = "yunet_sface"

class SearchCandidate(BaseModel):
    url: str
    source: str
    thumbnail_url: Optional[str] = None
    image_url: Optional[str] = None  # direct image URL if different from url
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VerificationResult(BaseModel):
    is_match: bool
    confidence_score: float
    candidate: SearchCandidate
    face_details: Optional[FaceResult] = None
    matched_face_index: Optional[int] = None
    number_of_faces: int = 0
    threshold: float = 0.0
    pass_reason: str = ""
    artifact_path: str = ""
    combined_rank_score: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EvidenceManifest(BaseModel):
    schema_version: str = "1.0"
    created_at_utc: str
    pipeline_version: str = "1.0.0"
    face_model: str
    input_sha256: str
    discovered_source_url: str
    discovered_source: str
    discovered_title: str
    discovered_image_sha256: str
    search_provider: str = "SerpApi Google Lens"
    search_request_id: Optional[str] = None
    face_similarity: float
    verification_threshold: float
    matched_face_index: Optional[int] = None
    result_status: str

class PipelineEvent(BaseModel):
    event_type: str
    status: str
    message: str
    duration_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None

class PipelineResult(BaseModel):
    run_id: str
    status: str
    # Machine-readable code for *why* status is a halting value (e.g. "NO_FACE_DETECTED",
    # "BELOW_MATCH_THRESHOLD"), distinct from `status` so the UI/logs can show a precise
    # reason without overloading the terminal-state string itself.
    failure_reason: Optional[str] = None
    events: List[PipelineEvent]
    reference_face: Optional[FaceResult] = None
    selected_candidate: Optional[SearchCandidate] = None
    verification_results: List[VerificationResult] = Field(default_factory=list)
    manifest_path: Optional[str] = None
    evidence_hash: Optional[str] = None
    media_hash: Optional[str] = None
    transaction_hash: Optional[str] = None
    chain_record: Optional[Dict[str, Any]] = None
    final_verified: bool = False

class BlockchainRecord(BaseModel):
    tx_hash: str
    block_number: int
    manifest_id: str
    chain_id: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DownloadedMedia(BaseModel):
    source_url: str
    final_url: str
    content_bytes: bytes
    sha256_hash: str
    content_type: str
    byte_size: int
