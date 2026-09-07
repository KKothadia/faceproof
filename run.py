import subprocess
import sys
import os

def main():
    """Run the local Streamlit application."""
    print("Starting FaceProof application...")
    
    app_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\nFaceProof stopped.")
    except Exception as e:
        print(f"Failed to start Streamlit: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
