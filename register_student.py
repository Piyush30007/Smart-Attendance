from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from database import initialize_database, upsert_student, validate_student_id
from haarcascade_utils import (
    align_face,
    create_face_detector,
    detect_faces,
    largest_face_detection,
)
from storage_paths import DATASET_DIR
DEFAULT_IMAGE_COUNT = 60
DEFAULT_SAMPLE_EVERY_FRAMES = 3
DEFAULT_AVERAGING_WINDOW = 2
REGISTER_FACE_SIZE = (160, 160)
DEFAULT_MIN_SAVE_GAP_FRAMES = 12
DEFAULT_MIN_FACE_DIFFERENCE = 8.0
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def import_registration_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Missing registration dependency. Install packages with `python -m pip install -r requirements-registration.txt`."
        ) from error

    return cv2, np


def slugify_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return cleaned.strip("_") or "Student"


def build_student_dir(student_id: str, student_name: str, dataset_dir: Path = DATASET_DIR) -> Path:
    return dataset_dir / f"{student_id}__{slugify_name(student_name)}"


def is_distinct_face(candidate_bgr: Any, reference_bgr: Any | None, cv2: Any, np: Any) -> bool:
    if reference_bgr is None:
        return True

    candidate_gray = cv2.cvtColor(cv2.resize(candidate_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(cv2.resize(reference_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    difference = float(np.mean(cv2.absdiff(candidate_gray, reference_gray)))
    return difference >= DEFAULT_MIN_FACE_DIFFERENCE


def extract_face_image(rgb_frame: Any, detection: Any, cv2: Any) -> Any | None:
    face_image = align_face(rgb_frame, detection, cv2)
    if face_image is not None and getattr(face_image, "size", 0) > 0:
        return cv2.resize(face_image, REGISTER_FACE_SIZE)

    if detection is None:
        return None

    top, right, bottom, left = detection.location
    frame_height, frame_width = rgb_frame.shape[:2]
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    pad_x = int(width * 0.15)
    pad_y = int(height * 0.15)

    crop_left = max(left - pad_x, 0)
    crop_top = max(top - pad_y, 0)
    crop_right = min(right + pad_x, frame_width)
    crop_bottom = min(bottom + pad_y, frame_height)

    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None

    face_crop = rgb_frame[crop_top:crop_bottom, crop_left:crop_right]
    if face_crop is None or getattr(face_crop, "size", 0) == 0:
        return None

    return cv2.resize(face_crop, REGISTER_FACE_SIZE)


def save_student_images_from_rgb_frames(
    student_id: str,
    student_name: str,
    rgb_frames: list[Any],
    dataset_dir: Path = DATASET_DIR,
) -> dict[str, Any]:
    cv2, np = import_registration_dependencies()
    initialize_database()

    if not student_id.strip() or not student_name.strip():
        raise RuntimeError("Student ID and student name are required.")
    if not validate_student_id(student_id):
        raise RuntimeError(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long."
        )
    if not rgb_frames:
        raise RuntimeError("No browser images were received for registration.")

    save_dir = build_student_dir(student_id.strip(), student_name.strip(), dataset_dir=dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{save_dir.name}__", dir=str(dataset_dir)))

    detector = create_face_detector()
    captured = 0
    last_saved_face_bgr = None

    for rgb_frame in rgb_frames:
        detections = detect_faces(detector, rgb_frame, min_size=40)
        primary_face = largest_face_detection(detections)
        if primary_face is None:
            continue

        face_image = extract_face_image(rgb_frame, primary_face, cv2)
        if face_image is None or face_image.size == 0:
            continue

        image_path = temp_dir / f"{captured:03d}.jpg"
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        if not is_distinct_face(face_bgr, last_saved_face_bgr, cv2, np):
            continue

        cv2.imwrite(str(image_path), face_bgr)
        captured += 1
        last_saved_face_bgr = face_bgr

    if captured == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "saved_dir": None,
            "captured": 0,
            "message": "No usable faces were found in the browser snapshots. Move closer to the camera, face forward, and use better lighting.",
        }

    if save_dir.exists():
        shutil.rmtree(save_dir)
    temp_dir.replace(save_dir)

    upsert_student(
        student_id=student_id.strip(),
        name=student_name.strip(),
        image_dir=str(save_dir),
        encoding_count=0,
    )

    return {
        "saved_dir": save_dir,
        "captured": captured,
        "message": f"Saved {captured} registration image(s) for {student_name}.",
    }


def capture_student_images(
    student_id: str,
    student_name: str,
    image_count: int = DEFAULT_IMAGE_COUNT,
    camera_index: int = 0,
    dataset_dir: Path = DATASET_DIR,
    sample_every_frames: int = DEFAULT_SAMPLE_EVERY_FRAMES,
    averaging_window: int = DEFAULT_AVERAGING_WINDOW,
    manual_mode: bool = False,
) -> Path | None:
    cv2, np = import_registration_dependencies()
    initialize_database()

    if not student_id.strip() or not student_name.strip():
        print("[ERROR] Student ID and student name are required.")
        return None
    if not validate_student_id(student_id):
        print(
            "[ERROR] Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long."
        )
        return None

    save_dir = build_student_dir(student_id.strip(), student_name.strip(), dataset_dir=dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{save_dir.name}__", dir=str(dataset_dir)))

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[ERROR] Could not open camera index {camera_index}")
        return None

    detector = create_face_detector()

    print(f"[INFO] Registering {student_name} ({student_id})")
    if manual_mode:
        print("[INFO] Manual capture is ON. Press 'c' to capture the detected face")
    else:
        print("[INFO] Auto capture is ON. Keep your face in frame and turn slightly for variety")
        print(
            f"[INFO] Sampling every {sample_every_frames} frame(s) and averaging {averaging_window} sample(s) per saved image"
        )
    print("[TIP] Look straight at the camera first, then turn slightly left and right.")
    print("[TIP] Use good lighting and keep your face clearly visible.")
    print("[INFO] Press 'q' to stop early")

    captured = 0
    frame_index = 0
    sample_buffer: deque[Any] = deque(maxlen=max(averaging_window, 1))
    last_saved_face_bgr = None
    last_saved_frame_index = -DEFAULT_MIN_SAVE_GAP_FRAMES

    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Failed to read from camera.")
            break

        frame_index += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect_faces(detector, rgb_frame, min_size=40)
        primary_face = largest_face_detection(detections)

        if primary_face is not None:
            top, right, bottom, left = primary_face.location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{student_name} | Captured: {captured}/{image_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Auto capture from live video" if not manual_mode else "Press c to capture, q to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Buffered: {len(sample_buffer)}/{sample_buffer.maxlen}" if not manual_mode else "Manual capture mode",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if primary_face is not None and not manual_mode and frame_index % max(sample_every_frames, 1) == 0:
            face_image = extract_face_image(rgb_frame, primary_face, cv2)
            if face_image is not None and face_image.size > 0:
                sample_buffer.append(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                if len(sample_buffer) == sample_buffer.maxlen:
                    averaged_face = np.mean(np.stack(list(sample_buffer), axis=0), axis=0).astype(np.uint8)
                    if frame_index - last_saved_frame_index < DEFAULT_MIN_SAVE_GAP_FRAMES:
                        sample_buffer.clear()
                        continue
                    if not is_distinct_face(averaged_face, last_saved_face_bgr, cv2, np):
                        sample_buffer.clear()
                        continue

                    image_path = temp_dir / f"{captured:03d}.jpg"
                    cv2.imwrite(str(image_path), averaged_face)
                    captured += 1
                    print(f"[INFO] Saved {image_path.name}")
                    sample_buffer.clear()
                    last_saved_face_bgr = averaged_face
                    last_saved_frame_index = frame_index

                    if captured >= image_count:
                        break

        cv2.imshow("Manual Student Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if manual_mode and key == ord("c"):
            if primary_face is None:
                print("[WARNING] No face detected. Try again with better lighting.")
                continue

            face_image = extract_face_image(rgb_frame, primary_face, cv2)
            if face_image is None or face_image.size == 0:
                print("[WARNING] Could not extract a usable face crop. Try keeping your face in better light.")
                continue

            image_path = temp_dir / f"{captured:03d}.jpg"
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            if not is_distinct_face(face_bgr, last_saved_face_bgr, cv2, np):
                print("[WARNING] Face is too similar to the last saved image. Change angle or expression slightly.")
                continue

            cv2.imwrite(str(image_path), face_bgr)
            captured += 1
            print(f"[INFO] Saved {image_path.name}")
            last_saved_face_bgr = face_bgr
            last_saved_frame_index = frame_index

            if captured >= image_count:
                break

    camera.release()
    cv2.destroyAllWindows()

    if captured == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[WARNING] No images captured.")
        return None

    if save_dir.exists():
        shutil.rmtree(save_dir)
    temp_dir.replace(save_dir)

    upsert_student(
        student_id=student_id.strip(),
        name=student_name.strip(),
        image_dir=str(save_dir),
        encoding_count=0,
    )

    print(f"[SUCCESS] Registration complete for {student_name}")
    print(f"[INFO] Images stored in {save_dir}")
    print("[NEXT STEP] Run `python encode_faces.py` to generate embeddings.")
    return save_dir


def prompt_for_registration() -> tuple[str, str, int, int]:
    student_id = input("Enter student ID: ").strip()
    student_name = input("Enter student name: ").strip()

    image_count_input = input(f"Images to capture [{DEFAULT_IMAGE_COUNT}]: ").strip()
    camera_index_input = input("Camera index [0]: ").strip()

    try:
        image_count = int(image_count_input) if image_count_input else DEFAULT_IMAGE_COUNT
        camera_index = int(camera_index_input) if camera_index_input else 0
    except ValueError as error:
        raise RuntimeError("Image count and camera index must be numbers.") from error

    return student_id, student_name, image_count, camera_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual student registration for Smart Attendance.")
    parser.add_argument("--student-id", type=str, default=None, help="Student ID to register.")
    parser.add_argument("--student-name", type=str, default=None, help="Student name to register.")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGE_COUNT, help="Number of face images to capture.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR, help="Dataset root where student folders are stored.")
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY_FRAMES, help="In auto mode, sample one face crop every N frames.")
    parser.add_argument("--average-window", type=int, default=DEFAULT_AVERAGING_WINDOW, help="In auto mode, average this many sampled crops into one saved image.")
    parser.add_argument("--manual", action="store_true", help="Use manual capture mode and press 'c' for each image.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    try:
        if args.student_id and args.student_name:
            capture_student_images(
                student_id=args.student_id,
                student_name=args.student_name,
                image_count=max(args.images, 1),
                camera_index=args.camera_index,
                dataset_dir=args.dataset,
                sample_every_frames=max(args.sample_every, 1),
                averaging_window=max(args.average_window, 1),
                manual_mode=args.manual,
            )
        else:
            student_id, student_name, image_count, camera_index = prompt_for_registration()
            capture_student_images(
                student_id=student_id,
                student_name=student_name,
                image_count=max(image_count, 1),
                camera_index=camera_index,
                dataset_dir=args.dataset,
                sample_every_frames=max(args.sample_every, 1),
                averaging_window=max(args.average_window, 1),
                manual_mode=args.manual,
            )
    except RuntimeError as error:
        print(f"[ERROR] {error}")
    except Exception as error:
        print(f"[ERROR] Unexpected registration failure: {error}")
