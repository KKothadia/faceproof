import cv2
import numpy as np
from typing import List
from src.schemas import FaceResult

def draw_faces(image: np.ndarray, faces: List[FaceResult]) -> np.ndarray:
    """Helper to visualize bounding boxes and landmarks on an image."""
    img_draw = image.copy()
    
    for face in faces:
        x, y, w, h = face.bbox
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Standard YuNet 5 points: right eye, left eye, nose, right mouth corner, left mouth corner
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (lx, ly) in enumerate(face.landmarks):
            color = colors[i % len(colors)]
            cv2.circle(img_draw, (lx, ly), 2, color, 2)
            
        text = f"{face.confidence:.2f} (Q:{face.quality_score:.2f})"
        cv2.putText(img_draw, text, (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
    return img_draw
