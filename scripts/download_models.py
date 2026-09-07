import os
import sys
import urllib.request

def download_models():
    """Download OpenCV Face models (YuNet and SFace).

    OpenCV 4.x: face_detection_yunet_2023mar.onnx (fixed-input, legacy DNN engine)
    OpenCV 5.x: face_detection_yunet_2023mar.onnx silently fails — use 2026may.onnx
                 OR install opencv-contrib-python==4.10.0.84 inside your venv.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    zoo = "https://github.com/opencv/opencv_zoo/raw/main/models"

    downloads = [
        (
            f"{zoo}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
        ),
        (
            f"{zoo}/face_detection_yunet/face_detection_yunet_2026may.onnx",
            os.path.join(models_dir, "face_detection_yunet_2026may.onnx"),
        ),
        (
            f"{zoo}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"),
        ),
    ]

    for url, path in downloads:
        name = os.path.basename(path)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  Already exists: {name} ({size:,} bytes)")
            continue
        print(f"  Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            print(f"  OK: {name} ({size:,} bytes)")
        except Exception as e:
            print(f"  FAILED: {name} — {e}")

    # Advise which model to use
    try:
        import cv2
        major = int(cv2.__version__.split(".")[0])
        if major >= 5:
            print()
            print("  *** OpenCV 5.x detected ***")
            print("  The 2023mar.onnx model silently returns no detections on OpenCV 5.x")
            print("  Recommended fix (choose one):")
            print("    A) Add to your .env:")
            print("         MODEL_YUNET_PATH=models/face_detection_yunet_2026may.onnx")
            print("    B) Downgrade OpenCV inside your venv:")
            print("         pip install 'opencv-contrib-python==4.10.0.84' --force-reinstall")
    except ImportError:
        pass


if __name__ == "__main__":
    download_models()
