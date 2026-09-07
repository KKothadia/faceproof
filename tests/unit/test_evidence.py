import os
import json
import pytest

from src.evidence.packager import EvidencePackager, verify_manifest, compute_sha256

def test_manifest_creation_and_hashing(tmp_path):
    run_id = "test_run_123"
    packager = EvidencePackager(run_id=run_id, artifact_base_dir=str(tmp_path))
    
    input_bytes = b"original_image_bytes"
    candidate_bytes = b"downloaded_candidate_bytes"
    
    manifest, original_hash = packager.create_manifest(
        input_bytes=input_bytes,
        candidate_bytes=candidate_bytes,
        face_model="opencv_yunet_sface",
        discovered_source_url="https://example.com",
        discovered_source="example.com",
        discovered_title="A Test Page",
        face_similarity=0.95,
        verification_threshold=0.90,
        result_status="Match",
        matched_face_index=0
    )
    
    run_dir = os.path.join(tmp_path, run_id)
    json_path = os.path.join(run_dir, "evidence.json")
    
    assert os.path.exists(json_path)
    assert os.path.exists(os.path.join(run_dir, "evidence.sha256"))
    assert os.path.exists(os.path.join(run_dir, "candidate.sha256"))
    
    # Acceptance: same manifest -> same hash
    assert verify_manifest(json_path, original_hash) is True
    
def test_tamper_evidence(tmp_path):
    run_id = "test_run_tamper"
    packager = EvidencePackager(run_id=run_id, artifact_base_dir=str(tmp_path))
    
    manifest, original_hash = packager.create_manifest(
        input_bytes=b"input",
        candidate_bytes=b"candidate",
        face_model="model",
        discovered_source_url="http://original.com",
        discovered_source="original",
        discovered_title="title",
        face_similarity=0.9,
        verification_threshold=0.8,
        result_status="Match"
    )
    
    run_dir = os.path.join(tmp_path, run_id)
    json_path = os.path.join(run_dir, "evidence.json")
    
    # Tamper with the JSON directly
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Alter one explicit manifest value
    data["discovered_source_url"] = "http://fake.com"
    
    # Write tampered dictionary back
    with open(json_path, "w", encoding="utf-8") as f:
        # Use simple dump (tampered)
        json.dump(data, f)
        
    # Acceptance: different field -> different hash
    # verify_manifest should now return False against the original hash
    assert verify_manifest(json_path, original_hash) is False

def test_different_candidate_bytes_hash(tmp_path):
    run_id = "test_run_candidate"
    packager = EvidencePackager(run_id=run_id, artifact_base_dir=str(tmp_path))
    
    _, hash1 = packager.create_manifest(
        input_bytes=b"input",
        candidate_bytes=b"candidate_A",
        face_model="model",
        discovered_source_url="http://original.com",
        discovered_source="original",
        discovered_title="title",
        face_similarity=0.9,
        verification_threshold=0.8,
        result_status="Match"
    )
    
    # Verify candidate.sha256 contains hash of candidate_A
    with open(os.path.join(tmp_path, run_id, "candidate.sha256"), "r") as f:
        cand_hash_A = f.read()
    assert cand_hash_A == compute_sha256(b"candidate_A")
    
    # Create another manifest with different bytes
    run_id2 = "test_run_candidate_2"
    packager2 = EvidencePackager(run_id=run_id2, artifact_base_dir=str(tmp_path))
    _, hash2 = packager2.create_manifest(
        input_bytes=b"input",
        candidate_bytes=b"candidate_B",
        face_model="model",
        discovered_source_url="http://original.com",
        discovered_source="original",
        discovered_title="title",
        face_similarity=0.9,
        verification_threshold=0.8,
        result_status="Match"
    )
    
    with open(os.path.join(tmp_path, run_id2, "candidate.sha256"), "r") as f:
        cand_hash_B = f.read()
        
    # Acceptance: different candidate bytes -> different media hash
    assert cand_hash_A != cand_hash_B
