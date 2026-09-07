# FaceProof

FaceProof is a local-only Python application designed to verify a physical face against social media and anchor the proof on the Base Sepolia blockchain.

## Architecture

The system consists of the following independently testable modules:

- **`src.face`**: Uses OpenCV YuNet and SFace for local face detection and feature extraction without relying on external cloud APIs.
- **`src.search`**: Performs a genuine reverse-image search via SerpApi Google Lens to find matching candidates online. No hardcoded target posts are used.
- **`src.verify`**: Compares the local face scan with the candidates retrieved from the search, scoring the matches.
- **`src.evidence`**: Constructs a cryptographically secure manifest containing the verification results.
- **`src.blockchain`**: Anchors the evidence manifest hash to the Base Sepolia public test blockchain using Web3.py.
- **`app`**: A local Streamlit UI for the end-user to interact with the system. No production deployment or web hosting is required.

## Constraints Adherence
- **Local-Only UI**: Uses Streamlit.
- **No LLM Dependency**: Relies on deterministic computer vision and search APIs at runtime.
- **Base Sepolia**: Serves as the public test blockchain for on-chain verification.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure your API keys (SerpApi Key, Base Sepolia RPC, Wallet Private Key).
5. Download local OpenCV models: `python scripts/download_models.py`
6. Deploy the Evidence Registry contract to Base Sepolia (Requires Testnet ETH): `python scripts/deploy_contract.py`
7. Copy the deployed contract address to `CONTRACT_ADDRESS` in `.env`.
8. Run the smoke test to verify connectivity: `python scripts/smoke_test.py`
9. Run the application UI: `python run.py`

## Testing
Run the offline unit test suite using pytest:
```bash
python -m pytest tests/unit/
```
