# FaceProof

FaceProof is a local-only Python application designed to verify a physical face against social media and anchor the proof on the Base Sepolia blockchain.

## Architecture

```text
 Uploaded face image
        |
        v
 [ src.face.FaceAnalyzer ]        OpenCV YuNet (detect) + SFace (encode)
        |  reference face + embedding
        |  --- if no face: pipeline halts, status=NO_FACE ---
        v
 [ src.search.SerpApiClient ]     Genuine reverse-image search (SerpApi -> Google Lens)
        |  candidate posts (url, source, thumbnail)
        |  --- if zero candidates: pipeline halts, status=NO_CANDIDATES ---
        v
 [ src.search.MediaDownloader ]   Downloads + validates each candidate's media
        |
        v
 [ src.verify.CandidateVerifier ] Independently re-detects + re-encodes each candidate
        |  image, compares embeddings via SFace cosine similarity
        |  --- if no candidate meets FACE_MATCH_COSINE_THRESHOLD: ---
        |  --- pipeline halts, status=NO_MATCH                    ---
        v
 [ src.evidence.EvidencePackager ] Canonical JSON manifest -> SHA-256 evidence hash
        |
        v
 [ src.blockchain.BlockchainClient ] anchorEvidence() on Base Sepolia, wait for receipt,
        |                            require status==1
        v
 [ read-back ]  verifyEvidence() reads the record back on-chain and the local hash is
                compared byte-for-byte against it -> status=SUCCESS only if they match
```

Every arrow above is a hard gate implemented in [`src/pipeline.py`](src/pipeline.py): each
stage only runs if the previous one produced a real, positive result, and the pipeline
records an explicit terminal status (`NO_FACE`, `NO_CANDIDATES`, `NO_MATCH`, `FAILED`,
`TAMPER_OR_CHAIN_MISMATCH`, `SUCCESS`) rather than ever guessing or defaulting to success.

The system consists of the following independently testable modules:

- **`src.face`**: Uses OpenCV YuNet and SFace for local face detection and feature extraction without relying on external cloud APIs.
- **`src.search`**: Performs a genuine reverse-image search via SerpApi Google Lens to find matching candidates online. No hardcoded target posts are used.
- **`src.verify`**: Compares the local face scan with the candidates retrieved from the search, scoring the matches.
- **`src.evidence`**: Constructs a cryptographically secure manifest containing the verification results.
- **`src.blockchain`**: Anchors the evidence manifest hash to the Base Sepolia public test blockchain using Web3.py.
- **`app`**: A local Streamlit UI for the end-user to interact with the system. No production deployment or web hosting is required.

### Search provider and its limitations

The search stage uses **SerpApi's Google Lens engine** as its single provider. This is a
genuine, programmatic reverse-image search (the image is uploaded and searched live -
nothing is scraped or hardcoded), which satisfies the project's "genuine web/social
search" requirement, but it comes with known limitations worth knowing about for the demo:

- **Single provider, no fallback.** If SerpApi is down or the account is rate-limited,
  the run fails with an explicit `SEARCH`/`FAILED` event rather than silently degrading
  or substituting a different result.
- **Result volume is capped.** `MAX_SEARCH_RESULTS` (default 20, see `.env.example`)
  bounds how many candidates are downloaded and face-verified per run, so a very large
  result set can't blow up runtime or amplify rate-limit exposure on the per-candidate
  media downloads.
- **429 (rate limited) is not retried.** A 429 fails the run immediately with a clear
  message rather than silently retrying against a provider that just asked to be backed
  off; re-run once the rate limit window resets.
- **Match quality depends on public indexing.** If the input face has no publicly
  indexed matching photo, `NO_CANDIDATES` (or `NO_MATCH` if unrelated look-alikes are
  returned) is the *correct* outcome, not a bug.

## Constraints Adherence
- **Local-Only UI**: Uses Streamlit.
- **No LLM Dependency**: Relies on deterministic computer vision and search APIs at runtime.
- **Base Sepolia**: Serves as the public test blockchain for on-chain verification.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in `SERPAPI_API_KEY` (get one at
   [serpapi.com](https://serpapi.com/manage-api-key)). Leave the blockchain and threshold
   variables as-is for now - later steps fill them in.
5. Download local OpenCV models: `python scripts/download_models.py`

### Testnet (Base Sepolia) setup
6. Generate or reuse a **throwaway** wallet and put its private key in `PRIVATE_KEY` in
   `.env`. Never use a wallet that holds real funds - this key signs live testnet
   transactions from your machine.
7. Fund that wallet with free Base Sepolia test ETH from a faucet, e.g.
   [Base Sepolia faucet](https://www.alchemy.com/faucets/base-sepolia). You only need a
   trace amount to cover gas.
8. Deploy the `EvidenceRegistry` contract: `python scripts/deploy_contract.py`. This
   compiles `contracts/EvidenceRegistry.sol`, deploys it to Base Sepolia (chain id
   `84532`), waits for the receipt, and prints the deployed address.
9. Copy that address into `CONTRACT_ADDRESS` in `.env`.
   If you ever need a different RPC endpoint, override `BASE_SEPOLIA_RPC_URL` - any
   Base Sepolia-compatible HTTPS RPC works (e.g. an Alchemy/Infura endpoint).

### Run
10. Run the smoke test to verify all connectivity end-to-end before using the UI:
    `python scripts/smoke_test.py`
11. Run the application UI: `python run.py`

## Configuration
All tunables live in `.env` (see `.env.example` for the full commented list: search
result cap, face detection/match thresholds, media size limit, model paths). `Config.validate()`
runs at the start of every pipeline execution and raises a `ConfigurationError` (surfaced
as a clean `FAILED` pipeline stage, not a crash) if a numeric setting is out of its valid
range - e.g. a threshold outside `(0, 1]`. Missing API keys are only warned about, since
the unit test suite and offline development don't need them.

### Tuning the match threshold
`FACE_MATCH_COSINE_THRESHOLD` (default `0.65`) is the single gate between "best guess"
and "verified match" - the pipeline never treats a top search result as verified without
this independently re-computed SFace similarity clearing it. If verification runs but
nothing passes, that is usually a sign to look at *why* before touching the threshold:
is the correct media actually being downloaded (check `artifacts/<run_id>/`), is the
candidate image quality/resolution too low, is the detected face poorly aligned, or is
this genuinely a non-match.

## Logging
Every pipeline run logs structured, single-line JSON events (stage, status, message,
duration, and diagnostic details such as best-candidate score vs. threshold) to both the
console and a rotating log file at `logs/faceproof.log` (created on first run, capped at
5MB x 3 backups, ignored by git). Secrets (the SerpApi key, the wallet private key) are
never included in any log line.

## Testing
```bash
# Offline unit tests (no network, no blockchain, no model files required)
python -m pytest tests/unit/

# Integration test: runs the full search -> verify -> evidence -> blockchain pipeline
# with only the network boundary (SerpApi HTTP calls) and blockchain RPC mocked -
# everything else (parsing, hashing, ranking) executes for real.
python -m pytest tests/integration/ -m "not live"

# Optional: a genuinely live SerpApi call (requires a real SERPAPI_API_KEY)
python -m pytest tests/integration/test_search_live.py --live-search
```
CI (`.github/workflows/ci.yml`) runs flake8 and both the unit and mocked-integration
suites on every push/PR.

## Known limitations
- Single search provider (SerpApi/Google Lens) - see "Search provider and its
  limitations" above. Identical repeated searches for the same input image are served
  from a local on-disk cache (`.cache/serpapi_search/`, keyed by image content hash) to
  save API quota during iterative demo runs; set `SEARCH_CACHE_ENABLED=false` to always
  hit SerpApi live.
- The search and verification stages run synchronously within the Streamlit request
  (not backgrounded/async). For a single demo image this completes in a few seconds;
  this was a deliberate choice over adding threading/async to Streamlit's rerun model,
  which would add real state-management risk for negligible benefit at this scale.
- Face match thresholds (`FACE_DETECTION_THRESHOLD`, `FACE_MATCH_COSINE_THRESHOLD`) are
  empirical defaults for the YuNet/SFace model pair, not universal constants; they may
  need re-tuning for very different lighting/pose conditions.
- Requires a funded Base Sepolia testnet wallet; without one, the pipeline still runs
  through search and verification but reports `FAILED` at the blockchain stage rather
  than fabricating a transaction.
