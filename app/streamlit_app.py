import streamlit as st
import tempfile
import os
import json
import atexit
import cv2
import pandas as pd
import numpy as np

from src.pipeline import FaceProofPipeline
from src.face.analyzer import FaceAnalyzer
from src.config import config
from src.evidence.packager import canonicalize_json, compute_sha256
from src.blockchain.client import BlockchainClient
from src.exceptions import BlockchainError, FaceDetectionError
from src.logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="FaceProof - HH Goa 2026", layout="wide", page_icon="🕵️")

@st.cache_resource(show_spinner="Loading face detection models...")
def get_face_analyzer():
    """
    Load the YuNet/SFace ONNX models once per Streamlit process instead of on every
    'RUN PIPELINE' click - model loading from disk is the most expensive fixed cost
    in each run. Returns None (rather than raising) if the model files aren't present,
    so a missing model surfaces as a clean pipeline FAILED state instead of crashing
    the Streamlit callback.
    """
    try:
        return FaceAnalyzer()
    except FaceDetectionError as e:
        st.session_state["_face_analyzer_error"] = str(e)
        return None

def _cleanup_temp_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def _stage_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded biometric image to a temp file, removing any previous upload's
    temp file first so we don't accumulate face images on disk across reruns/sessions.
    """
    _cleanup_temp_file(st.session_state.get("input_path"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        path = tmp_file.name
    atexit.register(_cleanup_temp_file, path)
    return path

def run_tamper_test():
    result = st.session_state.get("pipeline_result")
    if not result or not result.manifest_path:
        st.error("No valid manifest available for tamper test.")
        return
        
    st.markdown("### 🚨 TAMPER TEST")
    
    # 1. Load current canonical manifest from disk
    with open(result.manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    st.write("**Original field:** `discovered_title` = ", manifest_data.get("discovered_title", ""))
        
    # 2. Modify one harmless field in memory (Do not touch disk)
    manifest_data["discovered_title"] = "HACKED_TITLE_TAMPER_ATTACK"
    
    # 3. Recompute hash
    tampered_json = canonicalize_json(manifest_data)
    tampered_hash = compute_sha256(tampered_json.encode("utf-8"))
    
    st.write("**Modified field in memory:** `discovered_title` = ", manifest_data["discovered_title"])
    
    col1, col2, col3 = st.columns(3)
    
    # 4. Read original anchored hash from blockchain
    try:
        client = BlockchainClient()
        record = client.read_record(result.evidence_hash)
        on_chain_hash = record["evidenceHash"]
        
        col1.metric("LOCAL HASH", f"{tampered_hash[:10]}...{tampered_hash[-10:]}")
        col2.metric("ON-CHAIN HASH", f"{on_chain_hash[:10]}...{on_chain_hash[-10:]}")
        
        # 5. Show mismatch
        if tampered_hash.lower() != on_chain_hash.lower():
            col3.error("✗ MISMATCH")
        else:
            col3.success("✓ MATCH")
    except Exception as e:
        st.error(f"Failed to read from blockchain: {e}")

def main():
    st.title("FACEPROOF")
    st.subheader("Face Identification → Search → Verification → Blockchain")
    
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.write("### Upload Face Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            input_image_path = _stage_uploaded_file(uploaded_file)
            st.session_state.input_path = input_image_path

            st.image(input_image_path, caption="Input Face", use_container_width=True)

            if st.button("RUN PIPELINE", type="primary", use_container_width=True):
                st.session_state.pipeline_result = None

                with st.status("Executing Pipeline...", expanded=True) as status:
                    def _render_event_live(event):
                        # Called synchronously by the pipeline right after each stage
                        # completes, so the log fills in in real time instead of the UI
                        # freezing on one spinner until the entire run (search + every
                        # candidate download + verification + blockchain) is done.
                        dur = f"({event.duration_ms}ms)" if event.duration_ms else ""
                        if event.status == "SUCCESS":
                            st.write(f"✅ **{event.event_type}**: {event.message} {dur}")
                        else:
                            st.write(f"❌ **{event.event_type}**: {event.message} {dur}")

                    pipeline = FaceProofPipeline(face_analyzer=get_face_analyzer())
                    result = pipeline.run(input_image_path, on_event=_render_event_live)
                    st.session_state.pipeline_result = result

                    if result.status == "SUCCESS":
                        status.update(label="Pipeline Completed", state="complete", expanded=False)
                    else:
                        label = f"Pipeline Halted: {result.status}"
                        if result.failure_reason:
                            label += f" ({result.failure_reason})"
                        status.update(label=label, state="error", expanded=True)

    with col_output:
        result = st.session_state.get("pipeline_result", None)
        if not result:
            st.info("Upload an image and run the pipeline to see results here.")
            return
            
        st.write("### PIPELINE")

        # --- Stage states derived from pipeline result ---
        face_ok = result.reference_face is not None
        search_ran = result.status not in ("NO_FACE", "NO_FACE_DETECTED", "FAILED") or len(result.verification_results) > 0
        # More explicit: if NO_FACE the search never ran
        face_failed_statuses = {"NO_FACE", "NO_FACE_DETECTED"}
        face_failed = result.status in face_failed_statuses or (not face_ok and result.status != "SUCCESS")

        # --- Face Detection ---
        if face_ok:
            img = cv2.imread(st.session_state.input_path)
            bbox = result.reference_face.bbox
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.success("✓ FACE DETECTED")
            st.image(img_rgb, width=200)
            st.caption(f"Confidence: {result.reference_face.confidence:.3f}")
        else:
            st.error(f"✗ FACE ANALYSIS — {result.status}")

        # --- Search stage ---
        if face_failed:
            st.info("○ WEB SEARCH — SKIPPED (face detection failed)")
        elif result.status == "NO_CANDIDATES":
            st.error("✗ WEB SEARCH — No candidates returned")
            if result.failure_reason:
                st.caption(f"Reason code: `{result.failure_reason}`")
        else:
            st.success(f"✓ LIVE SEARCH (Provider: Google Lens)")
            st.write(f"Candidates Analyzed: {len(result.verification_results)}")

        if face_failed:
            st.info("○ MATCH VERIFICATION — SKIPPED")
            st.info("○ EVIDENCE — SKIPPED")
            st.info("○ BLOCKCHAIN — SKIPPED")
            st.markdown("---")
            st.error(f"Pipeline halted at FACE ANALYSIS. Status: **{result.status}**")
            if result.failure_reason:
                st.caption(f"Reason code: `{result.failure_reason}`")
            return

        st.markdown("---")
        st.write("### MATCH ANALYSIS")
        if len(result.verification_results) > 0:
            # Show table of all candidates with PASS/FAIL status
            table_data = []
            for i, vr in enumerate(result.verification_results):
                table_data.append({
                    "Rank": i + 1,
                    "Source": vr.candidate.source,
                    "Similarity": f"{vr.confidence_score:.3f}",
                    "Status": "✓ PASS" if vr.is_match else "✗ FAIL",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.warning("No candidates were verified.")
        # Final verification outcome
        if result.status == "SUCCESS":
            st.success("✓ VERIFIED MATCH")
            # Show selected match details
            if result.selected_candidate:
                best_match_vr = next((vr for vr in result.verification_results if vr.candidate.url == result.selected_candidate.url), None)
                if best_match_vr:
                    st.markdown("---")
                    st.write("### SELECTED MATCH")
                    if os.path.exists(best_match_vr.artifact_path):
                        st.image(best_match_vr.artifact_path, width=300)
                    st.write(f"**Source:** {result.selected_candidate.source}")
                    st.write(f"**URL:** {result.selected_candidate.url}")
                    st.write(f"**Similarity:** {best_match_vr.confidence_score:.3f}")
                    st.write(f"**Threshold:** {best_match_vr.threshold:.3f}")
        elif result.status == "NO_MATCH":
            st.error("✗ NO VERIFIED MATCH FOUND")
            if result.failure_reason:
                st.caption(f"Reason code: `{result.failure_reason}`")
            # Show best candidate for diagnostics
            if result.verification_results:
                best_vr = result.verification_results[0]
                st.markdown("---")
                st.write("**BEST CANDIDATE — NOT VERIFIED**")
                st.write(f"Source: {best_vr.candidate.source}")
                st.write(f"URL: {best_vr.candidate.url}")
                st.write(f"Similarity: {best_vr.confidence_score:.3f}")
                st.write(f"Threshold: {best_vr.threshold:.3f}")
        else:
            st.error(f"Pipeline execution halted early. Final Status: {result.status}")
            if result.failure_reason:
                st.caption(f"Reason code: `{result.failure_reason}`")
        st.markdown("---")
        st.write("### EVIDENCE")
        if result.evidence_hash:
            st.code(f"Evidence SHA-256: {result.evidence_hash}")
            st.code(f"Media SHA-256: {result.media_hash}")
            
        st.markdown("---")
        st.write("### BLOCKCHAIN")
        if result.chain_record:
            st.write(f"**Network:** Base Sepolia")
            st.write(f"**Contract:** {config.CONTRACT_ADDRESS}")
            st.write(f"**Transaction:** {result.transaction_hash}")
            
            if result.final_verified:
                st.success("✓ ON-CHAIN HASH MATCHES")
            else:
                st.error("✗ HASH MISMATCH")
                
        # Tamper test button
        st.markdown("---")
        if st.button("TAMPER TEST", type="secondary"):
            run_tamper_test()

if __name__ == "__main__":
    main()
