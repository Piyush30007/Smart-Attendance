from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FaceDetectionRecord:
    location: tuple[int, int, int, int]
    score: float
    keypoints: dict[str, tuple[float, float]]
    embedding: Any | None = None


@dataclass
class InsightFaceDetector:
    app: Any
    min_detection_confidence: float


def create_face_detector(min_detection_confidence: float = 0.55) -> InsightFaceDetector:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise RuntimeError(
            "Missing InsightFace dependency. Install from requirements.txt first."
        ) from error

    last_error = None
    provider_options = [
        ["CPUExecutionProvider"],
        None,
    ]

    for providers in provider_options:
        try:
            if providers is None:
                detector = FaceAnalysis(name="buffalo_l")
            else:
                detector = FaceAnalysis(name="buffalo_l", providers=providers)

            detector.prepare(
                ctx_id=-1,
                det_thresh=float(min_detection_confidence),
                det_size=(640, 640),
            )

            return InsightFaceDetector(
                app=detector,
                min_detection_confidence=float(min_detection_confidence),
            )
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(
        "Failed to initialize InsightFace detector. "
        "Check that onnxruntime and insightface are installed and configured."
    ) from last_error


def detect_faces(detector: InsightFaceDetector, rgb_image: Any, min_size: int = 24) -> list[FaceDetectionRecord]:
    import cv2
    import numpy as np

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return []

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    raw_faces = detector.app.get(bgr_image)

    image_height, image_width = rgb_image.shape[:2]
    detections: list[FaceDetectionRecord] = []

    for face in raw_faces:
        try:
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            left, top, right, bottom = [int(round(float(v))) for v in bbox]
            left = max(left, 0)
            top = max(top, 0)
            right = min(right, image_width)
            bottom = min(bottom, image_height)

            if (right - left) < min_size or (bottom - top) < min_size:
                continue

            score = float(
                getattr(face, "det_score", None)
                or getattr(face, "score", 0.0)
                or 0.0
            )
            if score < detector.min_detection_confidence:
                continue

            keypoints: dict[str, tuple[float, float]] = {}
            raw_kps = getattr(face, "kps", None)
            if raw_kps is not None and len(raw_kps) >= 2:
                keypoints["left_eye"] = (float(raw_kps[0][0]), float(raw_kps[0][1]))
                keypoints["right_eye"] = (float(raw_kps[1][0]), float(raw_kps[1][1]))

            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                raw_embedding = getattr(face, "embedding", None)
                if raw_embedding is not None:
                    norm = float(np.linalg.norm(raw_embedding))
                    if norm > 0:
                        embedding = raw_embedding / norm

            detections.append(
                FaceDetectionRecord(
                    location=(top, right, bottom, left),
                    score=score,
                    keypoints=keypoints,
                    embedding=embedding,
                )
            )
        except Exception:
            continue

    return detections


def extract_embedding_from_face_crop(detector: InsightFaceDetector, rgb_image: Any):
    import cv2
    import numpy as np

    if rgb_image is None or getattr(rgb_image, "size", 0) == 0:
        return None

    recognition_model = detector.app.models.get("recognition")
    if recognition_model is None:
        return None

    try:
        target_width, target_height = recognition_model.input_size
        resized = cv2.resize(rgb_image, (target_width, target_height))
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        embedding = recognition_model.get_feat(bgr).flatten()
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            return None

        return embedding / norm
    except Exception:
        return None


def largest_face_detection(detections: list[FaceDetectionRecord] | None):
    if not detections:
        return None

    return max(
        detections,
        key=lambda item: (item.location[1] - item.location[3]) * (item.location[2] - item.location[0]),
    )


def align_face(
    rgb_image: Any,
    detection: FaceDetectionRecord | None,
    cv2_module: Any,
    output_size: tuple[int, int] = (160, 160),
):
    if detection is None:
        return None

    left_eye = detection.keypoints.get("left_eye")
    right_eye = detection.keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        top, right, bottom, left = detection.location
        face = rgb_image[top:bottom, left:right]
        if face is None or getattr(face, "size", 0) == 0:
            return None
        return cv2_module.resize(face, output_size)

    try:
        image_height, image_width = rgb_image.shape[:2]
        center = (
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0,
        )

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))

        rotation_matrix = cv2_module.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2_module.warpAffine(
            rgb_image,
            rotation_matrix,
            (image_width, image_height),
            flags=cv2_module.INTER_CUBIC,
            borderMode=cv2_module.BORDER_REPLICATE,
        )

        top, right, bottom, left = detection.location
        corners = [
            _transform_point(left, top, rotation_matrix),
            _transform_point(right, top, rotation_matrix),
            _transform_point(right, bottom, rotation_matrix),
            _transform_point(left, bottom, rotation_matrix),
        ]

        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        min_x = max(int(min(xs)), 0)
        max_x = min(int(max(xs)), image_width)
        min_y = max(int(min(ys)), 0)
        max_y = min(int(max(ys)), image_height)

        if max_x <= min_x or max_y <= min_y:
            return None

        face = rotated[min_y:max_y, min_x:max_x]
        if face is None or getattr(face, "size", 0) == 0:
            return None

        return cv2_module.resize(face, output_size)
    except Exception:
        return None


def _transform_point(x: float, y: float, rotation_matrix: Any) -> tuple[float, float]:
    new_x = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2]
    new_y = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2]
    return new_x, new_y
