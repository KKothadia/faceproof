import os
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline import FaceProofPipeline
from src.schemas import PipelineResult, SearchCandidate, FaceResult, VerificationResult
from src.exceptions import SearchError, FaceDetectionError

@pytest.fixture
def dummy_image_path(tmp_path, monkeypatch):
    """Real file + isolated cwd so os.makedirs/os.path.exists behave normally (not globally mocked)."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "dummy.jpg"
    path.write_bytes(b"not-a-real-jpeg")
    return str(path)

@pytest.fixture
def mock_dependencies(dummy_image_path):
    with patch("src.pipeline.FaceAnalyzer") as mock_face, \
         patch("src.pipeline.SerpApiClient") as mock_search, \
         patch("src.pipeline.CandidateVerifier") as mock_verify, \
         patch("src.pipeline.EvidencePackager") as mock_packager, \
         patch("src.pipeline.BlockchainClient") as mock_blockchain, \
         patch("src.pipeline.cv2.imread") as mock_imread, \
         patch("src.pipeline.cv2.imdecode") as mock_imdecode, \
         patch("builtins.open", new_callable=MagicMock) as mock_open:

        mock_imdecode.return_value = "dummy_image"

        # When context manager is entered, return the mock file object which returns bytes on read
        mock_file = MagicMock()
        mock_file.read.return_value = b"fakebytes"
        mock_open.return_value.__enter__.return_value = mock_file

        yield mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open

def test_pipeline_fails_on_no_face(mock_dependencies, dummy_image_path):
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies
    
    # Analyze returns no face
    mock_face.return_value.analyze_image.return_value = []
    
    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)
        
    assert result.status == "NO_FACE"
    assert result.failure_reason == "NO_FACE_DETECTED"
    # Ensure SerpApi search is NEVER called when face detection fails
    mock_search.return_value.search_local_image.assert_not_called()
    # Ensure blockchain is NEVER called when earlier stages fail
    mock_blockchain.return_value.anchor_evidence.assert_not_called()

def test_pipeline_fails_on_no_candidates(mock_dependencies, dummy_image_path):
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies
    
    mock_face.return_value.analyze_image.return_value = [FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9)]
    mock_search.return_value.search_local_image.return_value = []
    
    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)
        
    assert result.status == "NO_CANDIDATES"
    assert result.failure_reason == "ZERO_SEARCH_CANDIDATES"
    mock_blockchain.return_value.anchor_evidence.assert_not_called()

def test_pipeline_fails_on_no_match(mock_dependencies, dummy_image_path):
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies
    
    mock_face.return_value.analyze_image.return_value = [FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9)]
    mock_search.return_value.search_local_image.return_value = [SearchCandidate(url="http", source="src")]
    
    vr = VerificationResult(
        is_match=False, confidence_score=0.1, candidate=SearchCandidate(url="http", source="src"),
        number_of_faces=1, threshold=0.363, pass_reason="Low", artifact_path="path"
    )
    mock_verify.return_value.verify_candidates.return_value = [vr]
    
    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)
        
    assert result.status == "NO_MATCH"
    assert result.failure_reason == "BELOW_MATCH_THRESHOLD"
    mock_blockchain.return_value.anchor_evidence.assert_not_called()

def test_pipeline_success_flow(mock_dependencies, dummy_image_path):
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies
    
    mock_face.return_value.analyze_image.return_value = [FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9)]
    mock_search.return_value.search_local_image.return_value = [SearchCandidate(url="http", source="src")]
    
    vr = VerificationResult(
        is_match=True, confidence_score=0.9, candidate=SearchCandidate(url="http", source="src"),
        number_of_faces=1, threshold=0.363, pass_reason="High", artifact_path="path"
    )
    mock_verify.return_value.verify_candidates.return_value = [vr]
    
    class MockManifest:
        discovered_image_sha256 = "dummy_sha256"
        
    mock_packager.return_value.create_manifest.return_value = (MockManifest(), "manifest_hash")
    
    mock_blockchain.return_value.anchor_evidence.return_value = ("0xTxHash", "Success")
    mock_blockchain.return_value.verify_against_chain.return_value = (True, "Chain match")
    mock_blockchain.return_value.read_record.return_value = {"record": "test"}
    
    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)
        
    assert result.status == "SUCCESS"
    assert result.failure_reason is None
    assert result.final_verified is True
    assert result.transaction_hash == "0xTxHash"
    # Blockchain MUST be called exactly once
    mock_blockchain.return_value.anchor_evidence.assert_called_once_with(evidence_hash="manifest_hash", media_hash="dummy_sha256")

def test_pipeline_on_event_callback_fires_live_for_every_stage(mock_dependencies, dummy_image_path):
    """The Streamlit UI renders progress via this callback instead of waiting for run() to
    finish - it must fire once per emitted event, in order, with the real event objects."""
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies

    mock_face.return_value.analyze_image.return_value = []  # halts at FACE_ANALYSIS

    seen = []
    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path, on_event=lambda event: seen.append(event))

    assert [e.event_type for e in seen] == [e.event_type for e in result.events]
    assert seen[-1].event_type == "FACE_ANALYSIS"
    assert seen[-1].status == "FAILED"

def test_pipeline_on_event_callback_exception_does_not_break_pipeline(mock_dependencies, dummy_image_path):
    """A buggy UI callback must never take the pipeline down with it."""
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies
    mock_face.return_value.analyze_image.return_value = []

    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path, on_event=lambda event: 1 / 0)

    assert result.status == "NO_FACE"

def test_pipeline_fails_gracefully_when_search_client_misconfigured(mock_dependencies, dummy_image_path):
    """
    A missing/invalid SERPAPI_API_KEY must produce a clean FAILED status, not an
    unhandled exception out of the constructor (mirrors BlockchainClient's degrade-gracefully
    pattern - see src/pipeline.py FaceProofPipeline.__init__).
    """
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies

    mock_search.side_effect = SearchError("SerpApi API key is not configured or is set to default.")
    mock_face.return_value.analyze_image.return_value = [FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9)]

    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)

    assert result.status == "FAILED"
    mock_blockchain.return_value.anchor_evidence.assert_not_called()

def test_pipeline_fails_gracefully_when_face_models_missing(mock_dependencies, dummy_image_path):
    """
    Missing YuNet/SFace ONNX model files must produce a clean FAILED status (with a
    message pointing at scripts/download_models.py), not an unhandled exception - this
    mirrors the SerpApiClient/BlockchainClient degrade-gracefully pattern.
    """
    mock_face, mock_search, mock_verify, mock_packager, mock_blockchain, mock_open = mock_dependencies

    mock_face.side_effect = FaceDetectionError("Detector model not found at models/face_detection_yunet_2023mar.onnx")

    pipeline = FaceProofPipeline()
    result = pipeline.run(dummy_image_path)

    assert result.status == "FAILED"
    mock_search.return_value.search_local_image.assert_not_called()
    mock_blockchain.return_value.anchor_evidence.assert_not_called()
