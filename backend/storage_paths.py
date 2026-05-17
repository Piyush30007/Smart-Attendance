from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BASE_DIR))).expanduser()

DB_PATH = DATA_ROOT / "smart_attendance.db"
ATTENDANCE_CSV = DATA_ROOT / "attendance.csv"
DATASET_DIR = DATA_ROOT / "dataset"
ENCODINGS_DIR = DATA_ROOT / "encodings"
ENCODINGS_PATH = ENCODINGS_DIR / "face_encodings.pkl"


def ensure_runtime_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)
