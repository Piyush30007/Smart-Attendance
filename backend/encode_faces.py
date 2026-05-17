from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

from database import initialize_database, upsert_student
from haarcascade_utils import (
    EMBEDDING_DIM,
    create_face_detector,
    detect_faces,
    extract_embedding_from_face_crop,
    largest_face_detection,
)
from storage_paths import DATASET_DIR, ENCODINGS_PATH, ensure_runtime_dirs
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM


def import_encoder_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Missing encoder dependency. Install packages from requirements.txt before running encode_faces.py."
        ) from error

    return cv2, np


def parse_student_folder(folder_name: str) -> tuple[str, str]:
    if "__" in folder_name:
        student_id, student_name = folder_name.split("__", 1)
        return student_id.strip(), clean_name(student_name)

    match = re.match(r"^([A-Za-z0-9]+)[_-](.+)$", folder_name)
    if match:
        return match.group(1).strip(), clean_name(match.group(2))

    return folder_name.strip(), clean_name(folder_name)


def clean_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def generate_augmented_variants(rgb_image: Any, cv2: Any, np: Any) -> list[tuple[str, Any]]:
    variants: list[tuple[str, Any]] = [("original", rgb_image)]

    darker = cv2.convertScaleAbs(rgb_image, alpha=0.78, beta=-18)
    variants.append(("darker", darker))

    warmer = rgb_image.astype(np.float32).copy()
    warmer[:, :, 0] = np.clip(warmer[:, :, 0] * 1.10 + 8, 0, 255)
    warmer[:, :, 1] = np.clip(warmer[:, :, 1] * 0.98, 0, 255)
    warmer[:, :, 2] = np.clip(warmer[:, :, 2] * 0.90, 0, 255)
    variants.append(("warmer", warmer.astype(np.uint8)))

    darker_warmer = cv2.convertScaleAbs(warmer.astype(np.uint8), alpha=0.82, beta=-12)
    variants.append(("darker-warmer", darker_warmer))

    return variants


def load_existing_encodings(output_path: Path) -> dict[str, list[Any]]:
    _, np = import_encoder_dependencies()
    if not output_path.exists():
        return {}

    try:
        with output_path.open("rb") as file:
            data = pickle.load(file)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    sanitized: dict[str, list[Any]] = {}
    for student_id, encodings in data.items():
        if not isinstance(encodings, list):
            continue

        valid_encodings = []
        for encoding in encodings:
            array = np.asarray(encoding, dtype=np.float32).reshape(-1)
            if array.shape == (EXPECTED_EMBEDDING_DIM,):
                valid_encodings.append(array)

        if valid_encodings:
            sanitized[str(student_id)] = valid_encodings

    return sanitized


def find_target_student_folder(student_folders: list[Path], student_id: str) -> Path | None:
    for student_folder in student_folders:
        folder_student_id, _ = parse_student_folder(student_folder.name)
        if folder_student_id == student_id:
            return student_folder
    return None


def encode_student_folder(
    student_folder: Path,
    detector: object,
    cv2: object,
    np: object,
) -> tuple[str, str, list[Any], int, int]:
    student_id, student_name = parse_student_folder(student_folder.name)
    image_paths = sorted(
        path
        for path in student_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        return student_id, student_name, [], 0, 0

    student_encodings: list[Any] = []
    failed_images = 0

    print(f"[INFO] Encoding {student_name} ({student_id}) from {len(image_paths)} image(s)")
    if len(image_paths) < 5:
        print(f"[WARNING] Only {len(image_paths)} source image(s) found for {student_name}. Aim for 5-10 images in different lighting/positions.")

    for image_path in image_paths:
        try:
            bgr_image = cv2.imread(str(image_path))
            if bgr_image is None:
                print(f"  [WARNING] Could not read {image_path.name}")
                failed_images += 1
                continue

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            variant_embeddings = 0

            for _, variant_image in generate_augmented_variants(rgb_image, cv2, np):
                detections = detect_faces(detector, variant_image, min_size=36)
                primary_face = largest_face_detection(detections)

                embedding = primary_face.embedding if primary_face is not None else None
                if embedding is None:
                    crop_source = variant_image
                    if primary_face is not None:
                        top, right, bottom, left = primary_face.location
                        crop_source = variant_image[top:bottom, left:right]
                    embedding = extract_embedding_from_face_crop(detector, crop_source)

                if embedding is None:
                    continue

                student_encodings.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
                variant_embeddings += 1

            if variant_embeddings == 0:
                print(f"  [WARNING] No face -> {image_path.name}")
                failed_images += 1
        except Exception as error:
            print(f"  [ERROR] {image_path.name}: {error}")
            failed_images += 1

    return student_id, student_name, student_encodings, len(image_paths), failed_images


def generate_face_encodings(
    dataset_dir: Path = DATASET_DIR,
    output_path: Path = ENCODINGS_PATH,
    student_id: str | None = None,
    rebuild_all: bool = False,
) -> None:
    cv2, np = import_encoder_dependencies()
    ensure_runtime_dirs()
    initialize_database()
    detector = create_face_detector()

    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")

    student_folders = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not student_folders:
        raise RuntimeError(f"No student folders found in {dataset_dir}")

    encodings_by_id: dict[str, list[Any]] = {} if rebuild_all else load_existing_encodings(output_path)
    total_images = 0
    failed_images = 0
    students_encoded = 0

    if student_id:
        target_folder = find_target_student_folder(student_folders, student_id)
        if target_folder is None:
            raise RuntimeError(f"No dataset folder found for student ID {student_id}")
        student_folders = [target_folder]

    for student_folder in student_folders:
        current_student_id, student_name, student_encodings, image_count, student_failed_images = encode_student_folder(
            student_folder=student_folder,
            detector=detector,
            cv2=cv2,
            np=np,
        )

        if image_count == 0:
            if rebuild_all:
                print(f"[WARNING] No images found for {student_folder.name}")
            continue

        failed_images += student_failed_images
        total_images += len(student_encodings)

        if not student_encodings:
            print(f"[WARNING] No valid encodings generated for {student_folder.name}")
            continue

        encodings_by_id[current_student_id] = student_encodings
        upsert_student(
            student_id=current_student_id,
            name=student_name,
            image_dir=str(student_folder),
            encoding_count=len(student_encodings),
        )
        students_encoded += 1
        print(f"[SUCCESS] Saved {len(student_encodings)} Haar embedding(s) for {student_name}")

    if not encodings_by_id:
        raise RuntimeError("No encodings generated")

    if student_id and students_encoded == 0:
        raise RuntimeError(f"No valid encodings were generated for student {student_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(encodings_by_id, file)

    print("\n===== DONE =====")
    print("Students:", len(encodings_by_id))
    print("Encoded:", total_images)
    print("Failed:", failed_images)
    print("Saved to:", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate face embeddings from the dataset folder.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=ENCODINGS_PATH)
    parser.add_argument("--student-id", type=str, default=None)
    parser.add_argument("--rebuild-all", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_face_encodings(
        dataset_dir=args.dataset,
        output_path=args.output,
        student_id=args.student_id,
        rebuild_all=args.rebuild_all,
    )
