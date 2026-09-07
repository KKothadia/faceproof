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
def test_read_record_matching_bytes(mock_account, mock_w3):
    mock_w3.to_bytes.side_effect = lambda hexstr: bytes.fromhex(hexstr.removeprefix("0x"))
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    ev_hex = "0d2294d1f8da3da3976e250dda452c820f5fa9dc56aca9f563cb50e95dad8639"
    md_hex = "511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51"
    
    ev_bytes = bytes.fromhex(ev_hex)
    md_bytes = bytes.fromhex(md_hex)
    
    mock_contract = MagicMock()
    mock_contract.functions.verifyEvidence.return_value.call.return_value = (
        ev_bytes,
        md_bytes,
        1788793322,
        "0xc375Da0BA71F33EF3b936BE0c6D9663eD76844b4"
    )
    client.contract = mock_contract
    client.w3.to_hex.return_value = "0x" + md_hex
    
    # Test 1 - matching bytes
    valid, msg = client.verify_against_chain(ev_hex, md_hex)
    assert valid is True
    assert "Verified on-chain" in msg

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_read_record_mismatching_stored_hash(mock_account, mock_w3):
    mock_w3.to_bytes.side_effect = lambda hexstr: bytes.fromhex(hexstr.removeprefix("0x"))
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    ev_hex = "0d2294d1f8da3da3976e250dda452c820f5fa9dc56aca9f563cb50e95dad8639"
    ev_bytes = bytes.fromhex(ev_hex)
    md_bytes = bytes.fromhex("511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51")
    
    wrong_ev_bytes = bytes.fromhex("1111111111111111111111111111111111111111111111111111111111111111")
    
    mock_contract = MagicMock()
    mock_contract.functions.verifyEvidence.return_value.call.return_value = (
        wrong_ev_bytes,
        md_bytes,
        1788793322,
        "0xc375Da0BA71F33EF3b936BE0c6D9663eD76844b4"
    )
    client.contract = mock_contract
    
    # Test 2 - mismatching stored hash
    valid, msg = client.verify_against_chain(ev_hex, "511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51")
    assert valid is False
    assert "Stored evidence hash mismatch" in msg

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_read_record_zero_timestamp(mock_account, mock_w3):
    mock_w3.to_bytes.side_effect = lambda hexstr: bytes.fromhex(hexstr.removeprefix("0x"))
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    ev_hex = "0d2294d1f8da3da3976e250dda452c820f5fa9dc56aca9f563cb50e95dad8639"
    ev_bytes = bytes.fromhex(ev_hex)
    md_bytes = bytes.fromhex("511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51")
    
    mock_contract = MagicMock()
    mock_contract.functions.verifyEvidence.return_value.call.return_value = (
        ev_bytes,
        md_bytes,
        0,
        "0xc375Da0BA71F33EF3b936BE0c6D9663eD76844b4"
    )
    client.contract = mock_contract
    
    # Test 3 - zero timestamp
    valid, msg = client.verify_against_chain(ev_hex, "511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51")
    assert valid is False
    assert "not found on chain (zero timestamp)" in msg

@patch("src.blockchain.client.Web3")
@patch("src.blockchain.client.Account")
def test_read_record_optional_0x_prefix(mock_account, mock_w3):
    mock_w3.to_bytes.side_effect = lambda hexstr: bytes.fromhex(hexstr.removeprefix("0x"))
    client = BlockchainClient(private_key="dummy_valid_key", contract_address="0x123")
    
    ev_hex = "0d2294d1f8da3da3976e250dda452c820f5fa9dc56aca9f563cb50e95dad8639"
    md_hex = "511c448fdb056175bb2bd7af9e1758e007e627782378fb6b4ce17db10d32bd51"
    
    ev_bytes = bytes.fromhex(ev_hex)
    md_bytes = bytes.fromhex(md_hex)
    
    mock_contract = MagicMock()
    mock_contract.functions.verifyEvidence.return_value.call.return_value = (
        ev_bytes,
        md_bytes,
        1788793322,
        "0xc375Da0BA71F33EF3b936BE0c6D9663eD76844b4"
    )
    client.contract = mock_contract
    client.w3.to_hex.return_value = "0x" + md_hex
    
    # Test 4 - optional 0x prefix
    # Call with 0x prefix for evidence and media hashes
    valid_with_prefix, msg_with_prefix = client.verify_against_chain("0x" + ev_hex, "0x" + md_hex)
    
    assert valid_with_prefix is True
    assert "Verified on-chain" in msg_with_prefix
    
    # Contract should be called with identical exact bytes32 representation
    mock_contract.functions.verifyEvidence.assert_called_with(ev_bytes)
