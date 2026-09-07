import pytest
from unittest.mock import patch, MagicMock

from src.blockchain.client import BlockchainClient
from src.exceptions import BlockchainError

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_missing_private_key(mock_account, mock_w3):
    with pytest.raises(BlockchainError, match="Private key is not configured"):
        BlockchainClient(private_key="your_wallet_private_key")

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_validate_wrong_network(mock_account, mock_w3):
    mock_instance = mock_w3.return_value
    mock_instance.is_connected.return_value = True
    mock_instance.eth.chain_id = 1 # Simulate mainnet instead of Base Sepolia (84532)
    
    client = BlockchainClient(private_key="dummy_valid_key")
    with pytest.raises(BlockchainError, match="Wrong network"):
        client.validate_network()

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_insufficient_balance(mock_account, mock_w3):
    mock_instance = mock_w3.return_value
    mock_instance.is_connected.return_value = True
    mock_instance.eth.chain_id = 84532
    mock_instance.eth.get_balance.return_value = 0 # 0 wei
    mock_instance.from_wei.return_value = 0.0 # 0 ETH
    
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    with pytest.raises(BlockchainError, match="has 0 balance"):
        client.anchor_evidence("hash1", "hash2")

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_failed_transaction(mock_account, mock_w3):
    mock_instance = mock_w3.return_value
    mock_instance.is_connected.return_value = True
    mock_instance.eth.chain_id = 84532
    mock_instance.eth.get_balance.return_value = int(1e18) # 1 ETH
    
    # Mock receipt status = 0 (reverted transaction)
    mock_instance.eth.wait_for_transaction_receipt.return_value = {'status': 0}
    mock_instance.to_hex.return_value = "0xdeadbeef"
    
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    # Mock contract functions to avoid deep Web3 mock wiring
    client.contract = MagicMock()
    
    with pytest.raises(BlockchainError, match="Transaction reverted on chain"):
        # The 64 character strings simulate bytes32 hashes without 0x
        client.anchor_evidence("a"*64, "b"*64)

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_verify_mismatch(mock_account, mock_w3):
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    # Mock read_record directly to test verification logic strictly
    with patch.object(client, "read_record") as mock_read:
        mock_read.return_value = {
            "evidenceHash": "abcd",
            "mediaHash": "efgh",
            "timestamp": 123456789,
            "submitter": "0xSubmitter"
        }
        
        # Test exact match
        valid, msg = client.verify_against_chain("abcd", "efgh")
        assert valid is True
        
        # Test evidence mismatch
        valid, msg = client.verify_against_chain("wrong", "efgh")
        assert valid is False
        assert "Evidence hash mismatch" in msg
        
        # Test media mismatch
        valid, msg = client.verify_against_chain("abcd", "wrong")
        assert valid is False
        assert "Media hash mismatch" in msg
