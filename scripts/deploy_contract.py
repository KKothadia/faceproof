import os
import sys
from solcx import compile_standard, install_solc
from web3 import Web3
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def deploy_contract():
    load_dotenv()
    
    rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
    private_key = os.getenv("PRIVATE_KEY")
    
    if not private_key or private_key == "your_wallet_private_key":
        logging.error("PRIVATE_KEY not configured. Please set it in .env")
        sys.exit(1)
        
    logging.info("Installing solc 0.8.24...")
    install_solc("0.8.24")
    
    logging.info("Compiling EvidenceRegistry.sol...")
    with open(os.path.join(os.path.dirname(__file__), "..", "contracts", "EvidenceRegistry.sol"), "r") as file:
        source_code = file.read()
        
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"EvidenceRegistry.sol": {"content": source_code}},
            "settings": {
                "outputSelection": {
                    "*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}
                }
            },
        },
        solc_version="0.8.24",
    )
    
    bytecode = compiled_sol["contracts"]["EvidenceRegistry.sol"]["EvidenceRegistry"]["evm"]["bytecode"]["object"]
    abi = compiled_sol["contracts"]["EvidenceRegistry.sol"]["EvidenceRegistry"]["abi"]
    
    logging.info("Connecting to RPC %s", rpc_url)
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    if not w3.is_connected():
        logging.error("Failed to connect to network.")
        sys.exit(1)
        
    chain_id = w3.eth.chain_id
    if chain_id != 84532:
        logging.error("Wrong network. Expected Chain ID 84532, got %d.", chain_id)
        sys.exit(1)
        
    account = w3.eth.account.from_key(private_key)
    logging.info("Deploying from address: %s", account.address)
    
    balance = w3.eth.get_balance(account.address)
    logging.info("Wallet balance: %f ETH", float(w3.from_wei(balance, 'ether')))
    
    if balance == 0:
        logging.error("Wallet has 0 ETH. Need Base Sepolia ETH to deploy.")
        sys.exit(1)
        
    EvidenceRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(account.address)
    
    logging.info("Building deployment transaction...")
    transaction = EvidenceRegistry.constructor().build_transaction(
        {
            "chainId": 84532,
            "gas": 3000000,
            "maxFeePerGas": w3.eth.gas_price,
            "maxPriorityFeePerGas": w3.eth.gas_price,
            "nonce": nonce,
        }
    )
    
    logging.info("Signing transaction...")
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
    
    logging.info("Sending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction) # type: ignore
    
    tx_hash_hex = w3.to_hex(tx_hash)
    logging.info("Transaction hash: %s", tx_hash_hex)
    logging.info("Waiting for receipt (this may take a minute)...")
    
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    if tx_receipt["status"] != 1:
        logging.error("Deployment failed! Transaction reverted.")
        sys.exit(1)
        
    contract_address = tx_receipt.contractAddress
    logging.info("=========================================")
    logging.info("Contract deployed successfully!")
    logging.info("Address: %s", contract_address)
    logging.info("Update your .env file with:")
    logging.info("CONTRACT_ADDRESS=%s", contract_address)
    logging.info("=========================================")

if __name__ == "__main__":
    deploy_contract()
