from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_NAME = "Facenet512"
DEFAULT_DETECTOR_BACKEND = "retinaface"


@dataclass
class FaceDetectionRecord:
    location: tuple[int, int, int, int]
    score: float
    keypoints: dict[str, tuple[float, float]]
    embedding: Any | None = None
    face_image: Any | None = None


@dataclass
class DeepFaceDetector:
    model_name: str
    detector_backend: str
    min_detection_confidence: float


def create_face_detector(
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    min_detection_confidence: float = 0.85,
) -> DeepFaceDetector:
    try:
        from deepface import DeepFace  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Missing DeepFace dependency. Install packages from requirements.txt first."
        ) from error

    return DeepFaceDetector(
        model_name=model_name,
        detector_backend=detector_backend,
        min_detection_confidence=float(min_detection_confidence),
    )


def detect_faces(detector: DeepFaceDetector, rgb_image: Any, min_size: int = 40) -> list[FaceDetectionRecord]:
    import cv2
    import numpy as np
    from deepface import DeepFace

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return []

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    try:
        raw_faces = DeepFace.extract_faces(
            img_path=bgr_image,
            detector_backend=detector.detector_backend,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        return []

    image_height, image_width = rgb_image.shape[:2]
    detections: list[FaceDetectionRecord] = []

    for raw_face in raw_faces:
        facial_area = raw_face.get("facial_area") or {}
        x = int(facial_area.get("x", 0))
        y = int(facial_area.get("y", 0))
        width = int(facial_area.get("w", 0))
        height = int(facial_area.get("h", 0))

        if width < min_size or height < min_size:
            continue

        left = max(x, 0)
        top = max(y, 0)
        right = min(left + width, image_width)
        bottom = min(top + height, image_height)

        if right <= left or bottom <= top:
            continue

        score = float(raw_face.get("confidence", 0.0) or 0.0)
        if score and score < detector.min_detection_confidence:
            continue

        face_image = _normalize_face_array(raw_face.get("face"))
        if face_image is None:
            face_image = rgb_image[top:bottom, left:right]

        embedding = extract_embedding_from_face_crop(detector, face_image)

        detections.append(
            FaceDetectionRecord(
                location=(top, right, bottom, left),
                score=score,
                keypoints={},
                embedding=embedding,
                face_image=face_image,
            )
        )

    return detections


def extract_embedding_from_face_crop(detector: DeepFaceDetector, rgb_image: Any):
    import cv2
    import numpy as np
    from deepface import DeepFace

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return None

    try:
        face_rgb = _normalize_face_array(rgb_image)
        if face_rgb is None:
            return None

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        representations = DeepFace.represent(
            img_path=face_bgr,
            model_name=detector.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="base",
            silent=True,
        )
    except Exception:
        return None

    if not representations:
        return None

    embedding = representations[0].get("embedding")
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None

    return vector / norm


def largest_face_detection(detections: list[FaceDetectionRecord] | None):
    if not detections:
        return None

    return max(
        detections,
        key=lambda item: (item.location[1] - item.location[3]) * (item.location[2] - item.location[0]),
    )


def align_face(rgb_image: Any, detection: FaceDetectionRecord | None, cv2_module: Any):
    if detection is None:
        return None

    if detection.face_image is not None and getattr(detection.face_image, "size", 0) > 0:
        return detection.face_image

    top, right, bottom, left = detection.location
    face = rgb_image[top:bottom, left:right]
    if face is None or getattr(face, "size", 0) == 0:
        return None

    return cv2_module.resize(face, (160, 160))


def _normalize_face_array(face_image: Any):
    import cv2
    import numpy as np

    if face_image is None:
        return None

    array = np.asarray(face_image)
    if array.size == 0:
        return None

    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max(initial=0) <= 1.0 else array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)

    if array.ndim != 3 or array.shape[2] != 3:
        return None

    return array
