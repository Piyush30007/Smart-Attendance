from __future__ import annotations

import argparse
import pickle
from collections import deque
from datetime import datetime
from typing import Any

from database import get_student, initialize_database, list_attendance, mark_attendance
from haarcascade_utils import (
    EMBEDDING_DIM,
    align_face,
    create_face_detector,
    detect_faces,
    extract_embedding_from_face_crop,
)
from storage_paths import ENCODINGS_PATH

DEFAULT_THRESHOLD = 0.55
DEFAULT_FRAME_SKIP = 2
SMOOTHING_WINDOW = 1
COOLDOWN_SECONDS = 5
MIN_CONFIDENCE_PERCENT = 68
EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM
# Detection resize width: the frame is resized to this width before detection.
# Coords are then mapped back to the ORIGINAL frame dimensions so the caller
# always receives boxes in the original (captured) coordinate space.
DETECTION_WIDTH = 320

_RUNTIME_DEPENDENCIES: tuple[Any, Any] | None = None


def import_runtime_dependencies() -> tuple[Any, Any]:
    global _RUNTIME_DEPENDENCIES
    if _RUNTIME_DEPENDENCIES is not None:
        return _RUNTIME_DEPENDENCIES
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Missing runtime dependency. Install packages from requirements.txt before running camera recognition."
        ) from error
    _RUNTIME_DEPENDENCIES = (cv2, np)
    return _RUNTIME_DEPENDENCIES


def load_encodings(encodings_path: Path = ENCODINGS_PATH) -> dict[str, list[Any]]:
    _, np = import_runtime_dependencies()
    if not encodings_path.exists():
        raise FileNotFoundError(
            f"Encodings file not found at {encodings_path}. Run encode_faces.py first."
        )
    with encodings_path.open("rb") as file:
        encodings = pickle.load(file)
    if not isinstance(encodings, dict) or not encodings:
        raise ValueError("Encoding file is empty or invalid.")

    sanitized: dict[str, list[Any]] = {}
    skipped = 0
    for student_id, known_encodings in encodings.items():
        if not isinstance(known_encodings, list):
            continue
        valid = []
        for enc in known_encodings:
            arr = np.asarray(enc, dtype=np.float32).reshape(-1)
            if arr.shape == (EXPECTED_EMBEDDING_DIM,):
                valid.append(arr)
            else:
                skipped += 1
        if valid:
            sanitized[str(student_id)] = valid

    if skipped:
        print(f"[WARNING] Skipped {skipped} stale embedding(s) with incompatible dimensions.")
    if not sanitized:
        raise ValueError(
            "Encoding file contains no valid Haar embeddings. Run `python encode_faces.py --rebuild-all`."
        )
    return sanitized


def apply_clahe(image: Any) -> Any:
    cv2, _ = import_runtime_dependencies()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def build_track_key(location: tuple[int, int, int, int], grid: int = 32) -> str:
    """
    Bin the face centre into a coarse grid cell so the key is stable across
    adjacent frames where the face moves a few pixels.
    """
    top, right, bottom, left = location
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    return f"{cx // grid}:{cy // grid}"


def find_best_match(
    face_encoding: Any,
    encodings_by_id: dict[str, list[Any]],
    threshold: float,
) -> tuple[str | None, float]:
    _, np = import_runtime_dependencies()
    best_id = None
    best_score = -1.0
    second_best = -1.0

    for student_id, known_encodings in encodings_by_id.items():
        scores = [
            float(np.dot(face_encoding, enc))
            for enc in known_encodings
            if getattr(enc, "shape", None) == getattr(face_encoding, "shape", None)
        ]
        if not scores:
            continue
        score = max(scores)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_id = student_id
        elif score > second_best:
            second_best = score

    if best_id is None or best_score < threshold or (best_score - second_best) < 0.04:
        return None, best_score
    return best_id, best_score


def process_frame(
    frame: Any,
    detector: Any,
    encodings_by_id: dict[str, list[Any]],
    threshold: float,
    recent_recognitions: dict[str, datetime],
    embedding_histories: dict[str, deque[Any]] | None = None,
) -> list[dict[str, object]]:
    """
    Process a single BGR frame and return detections.

    All returned bounding box coordinates are in the ORIGINAL frame's
    pixel space so the caller does not need to apply any scaling.
    """
    cv2, np = import_runtime_dependencies()
    orig_h, orig_w = frame.shape[:2]

    # --- Resize for detection (speed) ---
    scale = DETECTION_WIDTH / orig_w
    det_w = DETECTION_WIDTH
    det_h = int(orig_h * scale)
    small = cv2.resize(frame, (det_w, det_h))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_small = apply_clahe(rgb_small)

    histories = embedding_histories if embedding_histories is not None else {}
    detections = detect_faces(detector, rgb_small, min_size=20)

    if not detections:
        histories.clear()
        return []

    results: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for detection in sorted(
        detections,
        key=lambda d: (d.location[1] - d.location[3]) * (d.location[2] - d.location[0]),
        reverse=True,
    ):
        # Map coords back to original frame space
        s_top, s_right, s_bottom, s_left = detection.location
        top    = int(s_top    / scale)
        right  = int(s_right  / scale)
        bottom = int(s_bottom / scale)
        left   = int(s_left   / scale)

        # Clamp to frame bounds
        top    = max(0, min(top, orig_h))
        bottom = max(0, min(bottom, orig_h))
        left   = max(0, min(left, orig_w))
        right  = max(0, min(right, orig_w))

        orig_location = (top, right, bottom, left)
        track_key = build_track_key(orig_location)
        seen_keys.add(track_key)
        history = histories.setdefault(track_key, deque(maxlen=SMOOTHING_WINDOW))

        # Extract face crop from the SMALL frame for embedding
        face_image = align_face(rgb_small, detection, cv2)
        if face_image is None or getattr(face_image, "size", 0) == 0:
            history.clear()
            results.append({
                "label": "No face crop",
                "color": (0, 165, 255),
                "location": orig_location,
                "recognized": False,
            })
            continue

        embedding = extract_embedding_from_face_crop(detector, face_image)
        if embedding is None:
            history.clear()
            results.append({
                "label": "No embedding",
                "color": (0, 165, 255),
                "location": orig_location,
                "recognized": False,
            })
            continue

        enc = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(enc))
        if norm <= 0:
            history.clear()
            continue

        history.append(enc / norm)

        if len(history) < history.maxlen:
            results.append({
                "label": f"Stabilizing {len(history)}/{history.maxlen}",
                "color": (0, 165, 255),
                "location": orig_location,
                "recognized": False,
            })
            continue

        smoothed = np.mean(np.stack(list(history)), axis=0).astype(np.float32)
        s_norm = float(np.linalg.norm(smoothed))
        if s_norm <= 0:
            history.clear()
            continue
        smoothed /= s_norm

        student_id, score = find_best_match(smoothed, encodings_by_id, threshold)

        if student_id is None:
            history.clear()
            results.append({
                "label": "Unknown",
                "color": (0, 0, 255),
                "location": orig_location,
                "recognized": False,
            })
            continue

        student = get_student(student_id)
        student_name = student["name"] if student else student_id
        confidence = max(0, min(int(score * 100), 100))

        if confidence < MIN_CONFIDENCE_PERCENT:
            history.clear()
            results.append({
                "label": "Unknown",
                "color": (0, 0, 255),
                "location": orig_location,
                "recognized": False,
            })
            continue

        now = datetime.now()
        last_seen = recent_recognitions.get(student_id)
        attendance_recorded = False
        attendance_message = ""

        if last_seen is None or (now - last_seen).total_seconds() >= COOLDOWN_SECONDS:
            att = mark_attendance(student_id=student_id)
            recent_recognitions[student_id] = now
            attendance_recorded = bool(att["recorded"])
            attendance_message = str(att["message"])
            if att["recorded"]:
                print(f"[ATTENDANCE] {att['message']}")

        results.append({
            "label": f"{student_name} ({confidence}%)",
            "color": (0, 255, 0),
            "location": orig_location,
            "recognized": True,
            "student_id": student_id,
            "student_name": student_name,
            "confidence": confidence,
            "attendance_recorded": attendance_recorded,
            "attendance_message": attendance_message,
        })

    # Clean up stale track histories
    for key in [k for k in histories if k not in seen_keys]:
        histories.pop(key, None)

    return results


def run_attendance(
    camera_index: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    process_every_n_frames: int = DEFAULT_FRAME_SKIP,
) -> None:
    cv2, _ = import_runtime_dependencies()
    initialize_database()
    encodings_by_id = load_encodings()
    recent_recognitions: dict[str, datetime] = {}
    embedding_histories: dict[str, deque[Any]] = {}
    detector = create_face_detector(min_detection_confidence=0.55)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        return

    print("[INFO] Smart Attendance started. Press 'q' to quit.")
    frame_count = 0
    latest: list[dict[str, object]] = []

    while True:
        ok, frame = camera.read()
        if not ok:
            print("[ERROR] Camera read failed.")
            break
        frame_count += 1
        if frame_count % max(process_every_n_frames, 1) == 0:
            latest = process_frame(
                frame=frame,
                detector=detector,
                encodings_by_id=encodings_by_id,
                threshold=threshold,
                recent_recognitions=recent_recognitions,
                embedding_histories=embedding_histories,
            )
        draw_detections(frame, latest)
        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def draw_detections(frame: Any, detections: list[dict[str, object]]) -> None:
    cv2, _ = import_runtime_dependencies()
    for d in detections:
        top, right, bottom, left = d["location"]
        color = d["color"]
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, max(top - 30, 0)), (right, top), color, cv2.FILLED)
        cv2.putText(frame, str(d["label"]), (left + 6, max(top - 8, 18)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


def print_attendance_report(date_value: str | None = None) -> None:
    initialize_database()
    attendance_date = date_value or datetime.now().strftime("%Y-%m-%d")
    rows = list_attendance(attendance_date=attendance_date)
    if not rows:
        print(f"[INFO] No attendance records for {attendance_date}")
        return
    print(f"\n{'='*72}\nATTENDANCE REPORT - {attendance_date}\n{'='*72}")
    print(f"{'Student ID':<14}{'Student Name':<24}{'Time':<12}{'Status':<10}")
    print("-" * 72)
    for row in rows:
        print(f"{row['student_id']:<14}{row['student_name']:<24}{row['time']:<12}{row['status']:<10}")
    print(f"-{'-'*71}\nTotal present: {len(rows)}\n{'='*72}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart attendance main entry point.")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.view:
        print_attendance_report(date_value=args.date)
    else:
        try:
            run_attendance(
                camera_index=args.camera_index,
                threshold=args.threshold,
                process_every_n_frames=max(args.frame_skip, 1),
            )
        except RuntimeError as err:
            print(f"[ERROR] {err}")
