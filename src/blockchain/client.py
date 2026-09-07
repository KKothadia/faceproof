import logging
import time
from web3 import Web3
from web3.exceptions import Web3Exception
from eth_account import Account
from typing import Dict, Any, Tuple, Optional

from src.config import config
from src.exceptions import BlockchainError

logger = logging.getLogger(__name__)

EVIDENCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"}
        ],
        "name": "anchorEvidence",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"}],
        "name": "verifyEvidence",
        "outputs": [
            {
                "components": [
                    {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
                    {"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"},
                    {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
                    {"internalType": "address", "name": "submitter", "type": "address"}
                ],
                "internalType": "struct EvidenceRegistry.Record",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class BlockchainClient:
    def __init__(self, rpc_url: str = None, private_key: str = None, contract_address: str = None):
        self.rpc_url = rpc_url or config.BASE_SEPOLIA_RPC_URL
        self.private_key = private_key or config.PRIVATE_KEY
        self.contract_address = contract_address or config.CONTRACT_ADDRESS
        
        if not self.private_key or self.private_key == "your_wallet_private_key":
            raise BlockchainError("Private key is not configured.")
            
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self.account = Account.from_key(self.private_key)
        except Exception as e:
            # Mask exception details to prevent leaking private key
            logger.error("Failed to initialize Web3 client.")
            raise BlockchainError("Failed to initialize Web3 client.")
            
        if self.contract_address and self.contract_address != "your_deployed_contract_address":
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=EVIDENCE_REGISTRY_ABI
            )
        else:
            self.contract = None

    def validate_network(self) -> None:
        """Require exact chain ID 84532 (Base Sepolia) before interacting."""
        try:
            if not self.w3.is_connected():
                raise BlockchainError("Cannot connect to RPC node.")
            chain_id = self.w3.eth.chain_id
            if chain_id != 84532:
                raise BlockchainError(f"Wrong network. Expected Chain ID 84532, got {chain_id}.")
        except Exception as e:
            raise BlockchainError(f"Network validation failed: {str(e)}")
            
    def get_wallet_address(self) -> str:
        """Return the public address derived from the private key."""
        return self.account.address
        
    def get_balance(self) -> float:
        """Return wallet balance in ETH."""
        try:
            balance_wei = self.w3.eth.get_balance(self.account.address)
            return float(self.w3.from_wei(balance_wei, "ether"))
        except Exception as e:
            raise BlockchainError(f"Failed to fetch balance: {str(e)}")
            
    def anchor_evidence(self, evidence_hash: str, media_hash: str) -> Tuple[str, str]:
        """
        Anchor hashes to the contract, wait for receipt, and verify successful state.
        Returns the (transaction_hash, verification_message).
        """
        if not self.contract:
            raise BlockchainError("Contract address is not configured.")
            
        self.validate_network()
        
        balance = self.get_balance()
        if balance == 0:
            raise BlockchainError(f"Wallet {self.account.address} has 0 balance. Needs test ETH.")
            
        try:
            # Format hashes to bytes32
            ev_hash_bytes = Web3.to_bytes(hexstr=evidence_hash)
            md_hash_bytes = Web3.to_bytes(hexstr=media_hash)
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            tx = self.contract.functions.anchorEvidence(ev_hash_bytes, md_hash_bytes).build_transaction({
                'chainId': 84532,
                'gas': 2000000,
                'maxFeePerGas': self.w3.eth.gas_price,
                'maxPriorityFeePerGas': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction) # type: ignore
            tx_hash_hex = self.w3.to_hex(tx_hash)
            
            logger.info(f"Transaction sent: {tx_hash_hex}. Waiting for receipt...")
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] != 1:
                raise BlockchainError(f"Transaction reverted on chain. Tx: {tx_hash_hex}")
                
            # Read record back from the contract and verify
            is_valid, msg = self.verify_against_chain(evidence_hash, media_hash)
            if not is_valid:
                raise BlockchainError(f"Chain read-back verification failed: {msg}")
                
            return tx_hash_hex, msg
            
        except Exception as e:
            raise BlockchainError(f"Failed to anchor evidence: {str(e)}")
            
    def read_record(self, evidence_hash: str) -> Dict[str, Any]:
        """Read a record from the contract."""
        if not self.contract:
            raise BlockchainError("Contract address is not configured.")
            
        normalized = evidence_hash.removeprefix("0x").lower()
        if len(normalized) != 64:
            raise BlockchainError(f"Invalid evidence hash length: expected 64 hex characters, got {len(normalized)}")
            
        try:
            ev_hash_bytes = Web3.to_bytes(hexstr="0x" + normalized)
        except ValueError:
            raise BlockchainError(f"Invalid evidence hash format: {evidence_hash}")
            
        if len(ev_hash_bytes) != 32:
            raise BlockchainError(f"Invalid evidence hash byte length: expected 32, got {len(ev_hash_bytes)}")
            
        try:
            record = self.contract.functions.verifyEvidence(ev_hash_bytes).call()
        except Exception as e:
            # Differentiate reverting from general RPC errors
            if "revert" in str(e).lower():
                raise BlockchainError(f"Contract call reverted for evidence hash: {evidence_hash}")
            raise BlockchainError(f"RPC failure or contract error: {str(e)}")
            
        if record[2] == 0:
            raise BlockchainError(f"Evidence hash {evidence_hash} not found on chain (zero timestamp).")
            
        if record[0] != ev_hash_bytes:
            raise BlockchainError(f"Stored evidence hash mismatch for {evidence_hash}.")
            
        return {
            "evidenceHashBytes": record[0],
            "mediaHashBytes": record[1],
            "timestamp": record[2],
            "submitter": record[3]
        }

    def verify_against_chain(self, evidence_hash: str, expected_media_hash: str) -> Tuple[bool, str]:
        """
        Verify that the evidence and media hashes match the on-chain record exactly.
        """
        try:
            record = self.read_record(evidence_hash)
            
            norm_md = expected_media_hash.removeprefix("0x").lower()
            if len(norm_md) != 64:
                return False, f"Invalid expected media hash length: got {len(norm_md)}"
                
            try:
                expected_md_bytes = Web3.to_bytes(hexstr="0x" + norm_md)
            except ValueError:
                return False, f"Invalid expected media hash format: {expected_media_hash}"
                
            if record["mediaHashBytes"] != expected_md_bytes:
                chain_md_hex = self.w3.to_hex(record["mediaHashBytes"])
                return False, f"Media hash mismatch. Local: {norm_md}, Chain: {chain_md_hex}"
                
            return True, f"Verified on-chain. Anchored by {record['submitter']} at timestamp {record['timestamp']}."
        except BlockchainError as e:
            return False, str(e)
