from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FaceDetectionRecord:
    location: tuple[int, int, int, int]
    score: float
    keypoints: dict[str, tuple[float, float]]


def create_face_detector(min_detection_confidence: float = 0.7) -> Any:
    try:
        from mtcnn import MTCNN  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Missing MTCNN dependency. Install packages from requirements.txt before using face detection."
        ) from error

    detector = MTCNN()
    detector._min_detection_confidence = float(min_detection_confidence)
    return detector


def detect_faces(
    detector: Any,
    rgb_image: Any,
    min_size: int = 40,
) -> list[FaceDetectionRecord]:
    confidence = float(getattr(detector, "_min_detection_confidence", 0.7))
    detections = detector.detect_faces(
        rgb_image,
        box_format="xywh",
        output_type="json",
        postprocess=True,
        threshold_pnet=confidence,
        threshold_rnet=max(confidence, 0.7),
        threshold_onet=max(confidence, 0.8),
    )
    image_height, image_width = rgb_image.shape[:2]
    faces: list[FaceDetectionRecord] = []

    for detection in detections:
        box = detection.get("box")
        if not box or len(box) != 4:
            continue

        left, top, width, height = [int(value) for value in box]
        left = max(left, 0)
        top = max(top, 0)
        right = min(left + max(width, 1), image_width)
        bottom = min(top + max(height, 1), image_height)

        if (right - left) < min_size or (bottom - top) < min_size:
            continue

        raw_keypoints = detection.get("keypoints", {})
        keypoints = {
            name: (float(point[0]), float(point[1]))
            for name, point in raw_keypoints.items()
            if isinstance(point, (list, tuple)) and len(point) == 2
        }

        faces.append(
            FaceDetectionRecord(
                location=(top, right, bottom, left),
                score=float(detection.get("confidence", 0.0)),
                keypoints=keypoints,
            )
        )

    return faces


def largest_face_detection(
    detections: list[FaceDetectionRecord],
) -> FaceDetectionRecord | None:
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
    output_size: tuple[int, int] = (200, 200),
    padding_ratio: float = 0.2,
) -> Any | None:
    if detection is None:
        return None

    left_eye = detection.keypoints.get("left_eye")
    right_eye = detection.keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return None

    image_height, image_width = rgb_image.shape[:2]
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
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

    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    min_x = max(int(min(xs) - padding_ratio * box_width), 0)
    max_x = min(int(max(xs) + padding_ratio * box_width), image_width)
    min_y = max(int(min(ys) - padding_ratio * box_height), 0)
    max_y = min(int(max(ys) + padding_ratio * box_height), image_height)

    if max_x <= min_x or max_y <= min_y:
        return None

    aligned_face = rotated[min_y:max_y, min_x:max_x]
    if aligned_face.size == 0:
        return None

    return cv2_module.resize(aligned_face, output_size, interpolation=cv2_module.INTER_CUBIC)


def _transform_point(x: float, y: float, rotation_matrix: Any) -> tuple[float, float]:
    new_x = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2]
    new_y = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2]
    return new_x, new_y
