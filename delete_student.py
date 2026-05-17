from __future__ import annotations

import argparse
import csv
import pickle
import shutil
from pathlib import Path

from database import (
    ATTENDANCE_CSV,
    delete_student as delete_student_record,
    ensure_attendance_csv,
    validate_student_id,
)
from storage_paths import DATASET_DIR, ENCODINGS_PATH


def delete_student_dataset(student_id: str, dataset_dir: Path = DATASET_DIR) -> list[Path]:
    deleted_paths = []

    if not dataset_dir.exists():
        return deleted_paths

    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_student_id = folder.name.split("__", 1)[0]
        if folder_student_id == student_id:
            shutil.rmtree(folder)
            deleted_paths.append(folder)

    return deleted_paths


def delete_student_encodings(student_id: str, encodings_path: Path = ENCODINGS_PATH) -> bool:
    if not encodings_path.exists():
        return False

    with encodings_path.open("rb") as file:
        encodings = pickle.load(file)

    if student_id not in encodings:
        return False

    encodings.pop(student_id, None)
    with encodings_path.open("wb") as file:
        pickle.dump(encodings, file)

    return True


def delete_student_from_csv(student_id: str, csv_path: Path = ATTENDANCE_CSV) -> int:
    if not csv_path.exists():
        return 0

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if not rows:
        return 0

    header, *data_rows = rows
    kept_rows = [row for row in data_rows if len(row) < 2 or row[1] != student_id]
    deleted_rows = len(data_rows) - len(kept_rows)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(kept_rows)

    ensure_attendance_csv(csv_path)
    return deleted_rows


def delete_registered_student(student_id: str) -> None:
    if not validate_student_id(student_id):
        raise ValueError(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long."
        )

    result = delete_student_record(student_id)
    dataset_paths = delete_student_dataset(student_id)
    removed_from_encodings = delete_student_encodings(student_id)
    deleted_csv_rows = delete_student_from_csv(student_id)

    print(result["message"])
    if dataset_paths:
        for path in dataset_paths:
            print(f"[INFO] Deleted dataset folder: {path}")
    if removed_from_encodings:
        print("[INFO] Removed student from saved encodings.")
    if deleted_csv_rows:
        print(f"[INFO] Removed {deleted_csv_rows} attendance row(s) from attendance.csv.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete a registered student from Smart Attendance.")
    parser.add_argument("--student-id", required=True, help="Student ID to delete.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    delete_registered_student(args.student_id.strip())
