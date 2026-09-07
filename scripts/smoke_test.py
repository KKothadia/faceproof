import os
import sys
import requests
from web3 import Web3
from dotenv import load_dotenv

def smoke_test():
    print("Running Smoke Test...")
    load_dotenv()
    
    # 1. Configuration
    api_key = os.getenv("SERPAPI_API_KEY")
    rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
    private_key = os.getenv("PRIVATE_KEY")
    contract_addr = os.getenv("CONTRACT_ADDRESS")
    
    if not api_key or api_key == "your_serpapi_api_key":
        print("❌ SERPAPI_API_KEY is missing or invalid.")
        sys.exit(1)
        
    if not private_key or private_key == "your_wallet_private_key":
        print("❌ PRIVATE_KEY is missing or invalid.")
        sys.exit(1)
        
    if not contract_addr or contract_addr == "your_deployed_contract_address":
        print("❌ CONTRACT_ADDRESS is missing or invalid.")
        sys.exit(1)
        
    print("✅ Configuration loaded.")
    
    # 2. Model files
    models = ["models/face_detection_yunet_2023mar.onnx", "models/face_recognition_sface_2021dec.onnx"]
    for model in models:
        path = os.path.join(os.path.dirname(__file__), "..", model)
        if not os.path.exists(path):
            print(f"❌ Model file missing: {model}")
            sys.exit(1)
    print("✅ Local models found.")
    
    # 3. SerpApi Connectivity
    try:
        response = requests.get(f"https://serpapi.com/account?api_key={api_key}", timeout=10)
        if response.status_code == 200:
            print("✅ SerpApi connectivity successful.")
        else:
            print(f"❌ SerpApi check failed with status: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ SerpApi connection error: {e}")
        sys.exit(1)
        
    # 4. Base Sepolia Connectivity & Web3
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print("❌ Base Sepolia RPC connection failed.")
            sys.exit(1)
            
        print("✅ Base Sepolia connected.")
        
        account = w3.eth.account.from_key(private_key)
        print(f"✅ Wallet Address: {account.address}")
        
        balance_wei = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance_wei, "ether")
        print(f"✅ Wallet Balance: {balance_eth} ETH")
        
        if balance_eth == 0:
            print("❌ Wallet has 0 ETH. You need Base Sepolia ETH.")
            sys.exit(1)
            
        code = w3.eth.get_code(Web3.to_checksum_address(contract_addr))
        if code == b'' or code == b'0x':
            print(f"❌ No contract found at address {contract_addr}")
            sys.exit(1)
            
        print(f"✅ Contract verified at: {contract_addr}")
        
    except Exception as e:
        print(f"❌ Blockchain connectivity error: {e}")
        sys.exit(1)
        
    print("\n🚀 ALL SMOKE TESTS PASSED!")

if __name__ == "__main__":
    smoke_test()
