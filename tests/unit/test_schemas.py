from src.schemas import FaceResult, SearchCandidate, VerificationResult

def test_schemas_instantiation():
    """Test basic instantiation of Pydantic schemas."""
    face = FaceResult(
        bbox=[0, 0, 100, 100],
        landmarks=[[10, 10], [20, 20]],
        confidence=0.95
    )
    assert face.confidence == 0.95

    candidate = SearchCandidate(
        url="https://example.com/post/1",
        source="Google Lens"
    )
    assert candidate.url == "https://example.com/post/1"

    verification = VerificationResult(
        is_match=True,
        confidence_score=0.88,
        candidate=candidate,
        face_details=face
    )
    assert verification.is_match is True
