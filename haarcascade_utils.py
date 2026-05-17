from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EMBEDDING_DIM = 64 * 64


@dataclass
class FaceDetectionRecord:
    location: tuple[int, int, int, int]
    score: float
    keypoints: dict[str, tuple[float, float]]
    embedding: Any | None = None


@dataclass
class HaarCascadeDetector:
    face_cascade: Any


def create_face_detector(min_detection_confidence: float = 0.0) -> HaarCascadeDetector:
    import cv2

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return HaarCascadeDetector(face_cascade=face_cascade)


# ✅ SIMPLE FACE DETECTION
def detect_faces(detector: HaarCascadeDetector, rgb_image: Any, min_size: int = 40):
    import cv2

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return []

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    faces_raw = detector.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_size, min_size)
    )

    faces = []

    for (x, y, w, h) in faces_raw:
        top = y
        left = x
        bottom = y + h
        right = x + w

        faces.append(
            FaceDetectionRecord(
                location=(top, right, bottom, left),
                score=1.0,  # Haar doesn't give confidence
                keypoints={},  # no landmarks
                embedding=None
            )
        )

    return faces


# ✅ NO ALIGNMENT (just crop)
def align_face(rgb_image: Any, detection: FaceDetectionRecord | None, cv2_module: Any):
    if detection is None:
        return None

    top, right, bottom, left = detection.location

    face = rgb_image[top:bottom, left:right]

    if face is None or face.size == 0:
        return None

    return cv2_module.resize(face, (112, 112))


# ✅ BASIC EMBEDDING (flattened image)
def extract_embedding_from_face_crop(detector: HaarCascadeDetector, rgb_image: Any):
    import numpy as np
    import cv2

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return None

    try:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64))

        embedding = resized.flatten().astype("float32")

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        return embedding / norm

    except Exception:
        return None


def largest_face_detection(detections):
    if not detections:
        return None

    return max(
        detections,
        key=lambda item: (item.location[1] - item.location[3]) *
                         (item.location[2] - item.location[0]),
    )
