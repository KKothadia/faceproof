"""
Full-pipeline integration test: search -> verification -> evidence -> blockchain.

Only external network/hardware boundaries are mocked:
  - requests.post/get (stands in for the live SerpApi + candidate-media HTTP calls)
  - BlockchainClient (no funded Base Sepolia wallet available in CI)
  - FaceAnalyzer (no local ONNX model binaries available in CI)

Everything in between - SerpApiClient's parsing/dedup, MediaDownloader's validation and
hashing, CandidateVerifier's ranking, and EvidencePackager's canonical-hash computation -
runs for real, so this test catches wiring bugs the fully component-mocked unit tests
(tests/unit/test_pipeline.py) can't.
"""
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from requests import Response

from src.pipeline import FaceProofPipeline
from src.schemas import FaceResult


def _make_jpeg_bytes(size=10):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def input_image_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "input.jpg"
    path.write_bytes(_make_jpeg_bytes())
    return str(path)


def _mock_requests(candidate_media_bytes: bytes):
    """Returns (post_side_effect, get_side_effect) wired for one SerpApi search + one candidate download."""
    upload_resp = MagicMock(spec=Response)
    upload_resp.status_code = 200
    upload_resp.json.return_value = {"image_id": "img_123"}

    search_resp = MagicMock(spec=Response)
    search_resp.status_code = 200
    search_resp.json.return_value = {
        "visual_matches": [
            {"link": "https://social.example.com/post/1", "title": "A matching post", "thumbnail": "https://cdn.example.com/thumb.jpg"}
        ]
    }

    media_resp = MagicMock(spec=Response)
    media_resp.status_code = 200
    media_resp.headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(candidate_media_bytes))}
    media_resp.url = "https://cdn.example.com/thumb.jpg"
    media_resp.iter_content = MagicMock(return_value=[candidate_media_bytes])

    def post_side_effect(url, **kwargs):
        return upload_resp

    def get_side_effect(url, **kwargs):
        if "serpapi.com" in url:
            return search_resp
        return media_resp

    return post_side_effect, get_side_effect


def _reference_and_candidate_faces(match: bool):
    reference = FaceResult(
        bbox=[0, 0, 5, 5], landmarks=[[0, 0]] * 5, confidence=0.95,
        quality_score=0.9, feature_vector=[1.0, 0.0, 0.0], model_identifier="yunet_sface"
    )
    candidate = FaceResult(
        bbox=[0, 0, 5, 5], landmarks=[[0, 0]] * 5, confidence=0.9,
        quality_score=0.8, feature_vector=[1.0, 0.0, 0.0] if match else [0.0, 1.0, 0.0],
        model_identifier="yunet_sface"
    )
    return reference, candidate


@patch("src.pipeline.BlockchainClient")
@patch("src.pipeline.FaceAnalyzer")
@patch("requests.get")
@patch("requests.post")
def test_full_pipeline_reaches_blockchain_on_verified_match(
    mock_post, mock_get, mock_face_analyzer_cls, mock_blockchain_cls, input_image_path
):
    candidate_bytes = _make_jpeg_bytes()
    post_side_effect, get_side_effect = _mock_requests(candidate_bytes)
    mock_post.side_effect = post_side_effect
    mock_get.side_effect = get_side_effect

    reference_face, candidate_face = _reference_and_candidate_faces(match=True)
    analyzer = mock_face_analyzer_cls.return_value
    analyzer.analyze_image.side_effect = [[reference_face], [candidate_face]]
    analyzer.compare_features.return_value = 0.9  # above default 0.65 threshold

    blockchain = mock_blockchain_cls.return_value
    blockchain.anchor_evidence.return_value = ("0xabc123", "Anchored")

    captured = {}

    def fake_verify_against_chain(evidence_hash, expected_media_hash):
        captured["evidence_hash"] = evidence_hash
        captured["media_hash"] = expected_media_hash
        return True, "Verified on-chain"

    blockchain.verify_against_chain.side_effect = fake_verify_against_chain
    blockchain.read_record.return_value = {"evidenceHash": "x", "mediaHash": "y", "timestamp": 1, "submitter": "0xdead"}

    with patch("src.config.config.SERPAPI_API_KEY", "test_key"):
        pipeline = FaceProofPipeline()
        result = pipeline.run(input_image_path)

    assert result.status == "SUCCESS"
    assert result.final_verified is True
    assert result.transaction_hash == "0xabc123"
    assert result.selected_candidate.url == "https://social.example.com/post/1"

    # The hash chain must be consistent: what was anchored is exactly what evidence.json hashes to,
    # and exactly what read-back verification checked against.
    anchor_call_kwargs = blockchain.anchor_evidence.call_args.kwargs
    assert anchor_call_kwargs["evidence_hash"] == result.evidence_hash
    assert captured["evidence_hash"] == result.evidence_hash
    assert captured["media_hash"] == result.media_hash


@patch("src.pipeline.BlockchainClient")
@patch("src.pipeline.FaceAnalyzer")
@patch("requests.get")
@patch("requests.post")
def test_full_pipeline_stops_before_blockchain_when_no_candidate_meets_threshold(
    mock_post, mock_get, mock_face_analyzer_cls, mock_blockchain_cls, input_image_path
):
    candidate_bytes = _make_jpeg_bytes()
    post_side_effect, get_side_effect = _mock_requests(candidate_bytes)
    mock_post.side_effect = post_side_effect
    mock_get.side_effect = get_side_effect

    reference_face, candidate_face = _reference_and_candidate_faces(match=False)
    analyzer = mock_face_analyzer_cls.return_value
    analyzer.analyze_image.side_effect = [[reference_face], [candidate_face]]
    analyzer.compare_features.return_value = 0.1  # well below threshold

    blockchain = mock_blockchain_cls.return_value

    with patch("src.config.config.SERPAPI_API_KEY", "test_key"):
        pipeline = FaceProofPipeline()
        result = pipeline.run(input_image_path)

    assert result.status == "NO_MATCH"
    assert result.evidence_hash is None
    blockchain.anchor_evidence.assert_not_called()
