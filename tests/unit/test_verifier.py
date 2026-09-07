import pytest
from unittest.mock import MagicMock
import numpy as np
import cv2

from src.schemas import FaceResult, SearchCandidate, DownloadedMedia, VerificationResult
from src.verify.verifier import CandidateVerifier
from src.exceptions import VerificationError
from src.config import config

def create_dummy_media(valid: bool = True) -> DownloadedMedia:
    if valid:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        bytes_content = encoded.tobytes()
    else:
        bytes_content = b"invalid_bytes"
        
    return DownloadedMedia(
        source_url="http://dummy",
        final_url="http://dummy",
        content_bytes=bytes_content,
        sha256_hash="dummyhash",
        content_type="image/jpeg",
        byte_size=len(bytes_content)
    )

@pytest.fixture
def reference_face():
    return FaceResult(
        bbox=[0, 0, 10, 10],
        landmarks=[[0, 0]]*5,
        confidence=0.99,
        feature_vector=[0.1, 0.2, 0.3]
    )

@pytest.fixture
def candidates():
    return [
        SearchCandidate(url="http://exact.com", source="exact", metadata={"match_type": "exact_match", "position": 1}),
        SearchCandidate(url="http://visual.com", source="visual", metadata={"match_type": "visual_match", "position": 2}),
        SearchCandidate(url="http://noface.com", source="noface", metadata={"match_type": "visual_match", "position": 3})
    ]

def test_missing_reference_features(candidates):
    ref = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=None)
    mock_analyzer = MagicMock()
    verifier = CandidateVerifier(face_analyzer=mock_analyzer)
    
    with pytest.raises(VerificationError, match="must have a computed feature vector"):
        verifier.verify_candidates(ref, candidates, "tmp")

def test_verification_flow(reference_face, candidates):
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    
    # Mock media download (all valid)
    mock_downloader.download_candidate_media.return_value = create_dummy_media()
    
    # Setup specific faces for candidates
    face_exact = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.9, 0.9, 0.9])
    face_visual = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1, 0.1, 0.1])
    
    # Mock analyzer.analyze_image based on which candidate we are processing
    # We'll just return face lists.
    mock_analyzer.analyze_image.side_effect = [
        [face_exact],           # Candidate 1: 1 face
        [face_visual, face_exact], # Candidate 2: Multi-face, 2nd face is better
        []                      # Candidate 3: No face
    ]
    
    # Mock compare_features: returns 0.9 for exact, 0.2 for visual
    def mock_compare(f1, f2):
        if f2 == [0.9, 0.9, 0.9]: return 0.9
        if f2 == [0.1, 0.1, 0.1]: return 0.2
        return 0.0
    mock_analyzer.compare_features.side_effect = mock_compare
    
    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    verifier.threshold = 0.5 # Set threshold for predictable testing
    
    results = verifier.verify_candidates(reference_face, candidates, "tmp")
    
    assert len(results) == 3
    
    # Since they are sorted by combined_rank_score:
    # 1. Exact match candidate: score 0.9 + 0.05 bonus - 0.0001 (pos 1) = 0.9499 -> rank 1
    # 2. Visual match candidate: best face is face_exact (0.9) - 0.0002 (pos 2) = 0.8998 -> rank 2
    # 3. No face candidate: score 0.0 - 0.0003 = -0.0003 -> rank 3
    
    r1, r2, r3 = results
    
    assert r1.candidate.url == "http://exact.com"
    assert r1.is_match is True
    assert r1.confidence_score == 0.9
    
    assert r2.candidate.url == "http://visual.com"
    assert r2.is_match is True
    assert r2.confidence_score == 0.9
    assert r2.matched_face_index == 1 # Second face was better
    assert r2.number_of_faces == 2
    
    assert r3.candidate.url == "http://noface.com"
    assert r3.is_match is False
    assert r3.confidence_score == 0.0
    assert r3.number_of_faces == 0
    assert "No faces found" in r3.pass_reason

def test_skips_invalid_media(reference_face):
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    
    bad_cand = SearchCandidate(url="http://bad.com", source="bad")
    
    # Return None to simulate download failure
    mock_downloader.download_candidate_media.side_effect = [None, create_dummy_media(valid=False)]
    
    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    
    results = verifier.verify_candidates(reference_face, [bad_cand, bad_cand], "tmp")
    
    # Invalid candidates are silently rejected and not present in results
    assert len(results) == 0


def test_below_threshold_no_match():
    """Explicit: score below threshold => is_match=False"""
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    mock_downloader.download_candidate_media.return_value = create_dummy_media()

    candidate_face = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1, 0.2, 0.3])
    mock_analyzer.analyze_image.return_value = [candidate_face]
    mock_analyzer.compare_features.return_value = 0.3  # below threshold

    ref = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1, 0.2, 0.3])
    candidate = SearchCandidate(url="http://test.com", source="test")

    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    verifier.threshold = 0.65

    results = verifier.verify_candidates(ref, [candidate], "tmp")
    assert len(results) == 1
    assert results[0].is_match is False
    assert results[0].confidence_score == 0.3


def test_above_threshold_is_match():
    """Explicit: score >= threshold => is_match=True"""
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    mock_downloader.download_candidate_media.return_value = create_dummy_media()

    candidate_face = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1, 0.2, 0.3])
    mock_analyzer.analyze_image.return_value = [candidate_face]
    mock_analyzer.compare_features.return_value = 0.8  # above threshold

    ref = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1, 0.2, 0.3])
    candidate = SearchCandidate(url="http://test.com", source="test")

    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    verifier.threshold = 0.65

    results = verifier.verify_candidates(ref, [candidate], "tmp")
    assert len(results) == 1
    assert results[0].is_match is True
    assert results[0].confidence_score == 0.8


def test_best_candidate_below_threshold_but_another_passes():
    """The highest-ranked candidate may be below threshold while another passes."""
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    mock_downloader.download_candidate_media.return_value = create_dummy_media()

    face_low = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1])
    face_high = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.9])

    mock_analyzer.analyze_image.side_effect = [[face_low], [face_high]]

    def compare(f1, f2):
        if f2 == [0.1]: return 0.3  # below threshold
        if f2 == [0.9]: return 0.9  # above threshold
        return 0.0
    mock_analyzer.compare_features.side_effect = compare

    ref = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.5])
    cands = [
        SearchCandidate(url="http://low.com", source="low", metadata={"position": 1}),
        SearchCandidate(url="http://high.com", source="high", metadata={"position": 2}),
    ]

    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    verifier.threshold = 0.65

    results = verifier.verify_candidates(ref, cands, "tmp")
    passed = [r for r in results if r.is_match]
    assert len(passed) == 1
    assert passed[0].candidate.url == "http://high.com"
    assert results[0].candidate.url == "http://high.com"  # highest score = rank 1


def test_multi_face_candidate_uses_strongest_score():
    """With multiple faces in a candidate, the strongest score should be selected."""
    mock_analyzer = MagicMock()
    mock_downloader = MagicMock()
    mock_downloader.download_candidate_media.return_value = create_dummy_media()

    face_a = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.1])
    face_b = FaceResult(bbox=[0,0,20,20], landmarks=[[0,0]]*5, confidence=0.8, feature_vector=[0.9])
    mock_analyzer.analyze_image.return_value = [face_a, face_b]

    call_count = [0]
    def compare(f1, f2):
        call_count[0] += 1
        if f2 == [0.1]: return 0.3
        if f2 == [0.9]: return 0.85
        return 0.0
    mock_analyzer.compare_features.side_effect = compare

    ref = FaceResult(bbox=[0,0,10,10], landmarks=[[0,0]]*5, confidence=0.9, feature_vector=[0.5])
    candidate = SearchCandidate(url="http://multi.com", source="multi")

    verifier = CandidateVerifier(face_analyzer=mock_analyzer, downloader=mock_downloader)
    verifier.threshold = 0.65

    results = verifier.verify_candidates(ref, [candidate], "tmp")
    assert len(results) == 1
    assert results[0].confidence_score == 0.85  # strongest
    assert results[0].matched_face_index == 1  # second face (index 1)
    assert results[0].number_of_faces == 2
    assert results[0].is_match is True
