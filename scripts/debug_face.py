"""
debug_face.py — Standalone YuNet face detection diagnostic.
Does NOT call SerpApi, blockchain, or any pipeline code.

Usage:
    python scripts/debug_face.py --image <path_to_image>
"""
import argparse
import hashlib
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FaceProof YuNet diagnostic")
parser.add_argument("--image", required=True, help="Path to the image file to test")
parser.add_argument(
    "--model",
    default=os.getenv("MODEL_YUNET_PATH", "models/face_detection_yunet_2023mar.onnx"),
    help="Path to the YuNet ONNX model",
)
parser.add_argument("--threshold", type=float, default=None, help="Detection threshold (default: scan)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 60


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# 1. Image info
# ---------------------------------------------------------------------------
section("IMAGE INFO")
image_path = args.image
if not os.path.exists(image_path):
    print(f"ERROR: File not found: {image_path}")
    sys.exit(1)

file_size = os.path.getsize(image_path)
print(f"  Path       : {image_path}")
print(f"  File size  : {file_size:,} bytes ({file_size/1024:.1f} KB)")

# ---------------------------------------------------------------------------
# 2. EXIF orientation via Pillow
# ---------------------------------------------------------------------------
section("EXIF / IMAGE DECODING")
exif_applied = False
try:
    from PIL import Image, ImageOps

    pil_img = Image.open(image_path)
    original_size = pil_img.size
    exif_data = pil_img.getexif()
    exif_orientation = exif_data.get(274)  # tag 274 = Orientation
    print(f"  PIL mode (before transpose): {pil_img.mode}")
    print(f"  Original PIL size (WxH)    : {original_size}")
    print(f"  EXIF orientation tag       : {exif_orientation}")

    pil_img = ImageOps.exif_transpose(pil_img)
    transposed_size = pil_img.size
    exif_applied = transposed_size != original_size
    print(f"  EXIF transposition applied : {exif_applied} (new size: {transposed_size})")

    # Convert to BGR numpy for OpenCV
    pil_rgb = pil_img.convert("RGB")
    image = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    print(f"  Pillow → OpenCV BGR        : OK")
except ImportError:
    print("  Pillow not installed — falling back to cv2.imread()")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: cv2.imread returned None for {image_path}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Image array assertions
# ---------------------------------------------------------------------------
section("IMAGE ARRAY CHECKS")
assert image is not None, "Image is None"
assert isinstance(image, np.ndarray), "Image is not ndarray"
assert image.ndim == 3, f"Expected 3 dims, got {image.ndim}"
assert image.shape[2] == 3, f"Expected 3 channels, got {image.shape[2]}"
assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"

h, w = image.shape[:2]
print(f"  Shape      : {image.shape}  (HxWxC)")
print(f"  Width      : {w}")
print(f"  Height     : {h}")
print(f"  Channels   : {image.shape[2]}")
print(f"  dtype      : {image.dtype}")
print(f"  All checks : PASSED")

# ---------------------------------------------------------------------------
# 4. Model validation
# ---------------------------------------------------------------------------
section("MODEL VALIDATION")
model_path = args.model
print(f"  Path       : {model_path}")
if not os.path.exists(model_path):
    print(f"  ERROR: Model file not found at {model_path}")
    sys.exit(1)

model_size = os.path.getsize(model_path)
print(f"  File size  : {model_size:,} bytes")

with open(model_path, "rb") as f:
    header = f.read(4)
    f.seek(0)
    sha256 = hashlib.sha256(f.read()).hexdigest()

is_onnx = header[0] == 0x08
print(f"  Header hex : {header.hex()}")
print(f"  Valid ONNX : {is_onnx}")
print(f"  SHA256     : {sha256}")

with open(model_path, "r", errors="ignore") as f:
    first_line = f.readline().strip()
is_lfs = first_line.startswith("version https://git-lfs")
print(f"  Git LFS    : {is_lfs}")
if is_lfs:
    print("  ERROR: This is a Git LFS pointer, not a real ONNX file!")
    print("         Run: git lfs pull  OR  python scripts/download_models.py")
    sys.exit(1)
if not is_onnx:
    print("  ERROR: File does not appear to be a valid ONNX model.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. OpenCV / API check
# ---------------------------------------------------------------------------
section("OPENCV & API")
print(f"  cv2 version         : {cv2.__version__}")
print(f"  FaceDetectorYN      : {hasattr(cv2, 'FaceDetectorYN')}")
print(f"  FaceRecognizerSF    : {hasattr(cv2, 'FaceRecognizerSF')}")
print(f"  EXIF transposition  : {exif_applied}")

if not hasattr(cv2, "FaceDetectorYN"):
    print("  ERROR: cv2.FaceDetectorYN not available. Install opencv-contrib-python>=4.5.4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Detection at multiple thresholds
# ---------------------------------------------------------------------------
section("DETECTION SCAN (multiple thresholds)")
thresholds = [args.threshold] if args.threshold else [0.90, 0.80, 0.70, 0.60, 0.50]

any_detections = False
for thresh in thresholds:
    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(w, h),
            score_threshold=thresh,
            nms_threshold=0.3,
            top_k=5000,
        )
        detector.setInputSize((w, h))
        status, faces = detector.detect(image)
        n = 0 if faces is None else len(faces)
        scores = []
        if faces is not None:
            scores = [f"{float(f[14]):.4f}" for f in faces]
            any_detections = True
        print(f"  threshold={thresh:.2f}  |  detections={n}  |  scores={scores}")
    except Exception as e:
        print(f"  threshold={thresh:.2f}  |  ERROR: {e}")

# ---------------------------------------------------------------------------
# 7. Visual debug output
# ---------------------------------------------------------------------------
section("DEBUG IMAGE OUTPUT")
debug_dir = "artifacts/debug_face"
os.makedirs(debug_dir, exist_ok=True)

# Save normalized input
norm_path = os.path.join(debug_dir, "input_normalized.jpg")
cv2.imwrite(norm_path, image)
print(f"  Saved normalized input : {norm_path}")

# Save detection image using the lowest threshold that produced results
detect_path = os.path.join(debug_dir, "detections.jpg")
annotated = image.copy()

best_threshold = thresholds[-1]  # lowest threshold
try:
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(w, h),
        score_threshold=best_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[0:4].astype(int)
            score = float(face[14])
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{score:.2f}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        print(f"  Drew {len(faces)} detection(s) at threshold={best_threshold}")
    else:
        print(f"  No detections at threshold={best_threshold} — saving plain image")
except Exception as e:
    print(f"  Error drawing detections: {e}")

cv2.imwrite(detect_path, annotated)
print(f"  Saved detection image  : {detect_path}")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
section("SUMMARY")
print(f"  Image           : {image_path}")
print(f"  Size            : {w}x{h}")
print(f"  Model           : {os.path.basename(model_path)} ({model_size:,} bytes)")
print(f"  OpenCV          : {cv2.__version__}")
print(f"  EXIF rotated    : {exif_applied}")
print(f"  Any detections  : {any_detections}")
if not any_detections:
    print()
    print("  *** NO FACES DETECTED AT ANY THRESHOLD ***")
    print("  Possible causes:")
    print("    1. OpenCV 5.x + 2023mar.onnx incompatibility (MOST LIKELY)")
    print("       Fix: pip install 'opencv-contrib-python==4.10.0.84' --force-reinstall")
    print("    2. EXIF orientation causing upside-down/rotated image")
    print("    3. Corrupt ONNX model")
    print("    4. Image has no detectable face")
    print()
    print(f"  Check: {norm_path}")
    print(f"  And  : {detect_path}")
else:
    print()
    print("  Face(s) detected successfully.")
    print("  If production still shows NO_FACE, check the production threshold in .env")
    print("  (FACE_DETECTION_THRESHOLD=0.9 by default)")
