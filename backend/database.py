from __future__ import annotations

import csv
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from storage_paths import ATTENDANCE_CSV, DB_PATH, ensure_runtime_dirs

CSV_HEADER = ["date", "student_id", "student_name", "time", "status"]
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_runtime_dirs()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image_dir TEXT,
                encoding_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                source TEXT NOT NULL DEFAULT 'camera',
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(id),
                UNIQUE(student_id, attendance_date)
            );
            """
        )

    ensure_attendance_csv()


def ensure_attendance_csv(csv_path: Path = ATTENDANCE_CSV) -> None:
    ensure_runtime_dirs()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)


def validate_student_id(student_id: str) -> bool:
    return bool(STUDENT_ID_PATTERN.fullmatch(student_id.strip()))


def upsert_student(
    student_id: str,
    name: str,
    image_dir: str | None = None,
    encoding_count: int = 0,
    db_path: Path = DB_PATH,
) -> None:
    if not validate_student_id(student_id):
        raise ValueError(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long."
        )

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO students (id, name, image_dir, encoding_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                image_dir = excluded.image_dir,
                encoding_count = excluded.encoding_count,
                updated_at = excluded.updated_at
            """,
            (student_id, name, image_dir, encoding_count, now, now),
        )


def get_student(student_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT id, name, image_dir, encoding_count, created_at, updated_at FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()

    return dict(row) if row else None


def list_students(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, name, image_dir, encoding_count, created_at, updated_at
            FROM students
            ORDER BY name COLLATE NOCASE, id
            """
        ).fetchall()

    return [dict(row) for row in rows]


def delete_student(student_id: str, db_path: Path = DB_PATH) -> dict[str, Any]:
    initialize_database(db_path)

    if not validate_student_id(student_id):
        return {
            "deleted": False,
            "message": f"Student ID {student_id} is invalid.",
        }

    with get_connection(db_path) as connection:
        student = connection.execute(
            "SELECT id, name FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()

        if student is None:
            return {
                "deleted": False,
                "message": f"Student {student_id} was not found in the database.",
            }

        attendance_count = connection.execute(
            "SELECT COUNT(*) FROM attendance WHERE student_id = ?",
            (student_id,),
        ).fetchone()[0]

        connection.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        connection.execute("DELETE FROM students WHERE id = ?", (student_id,))

    return {
        "deleted": True,
        "student_name": student["name"],
        "deleted_attendance_rows": attendance_count,
        "message": f"Deleted student {student['name']} ({student_id}) from SQLite.",
    }


def mark_attendance(
    student_id: str,
    timestamp: datetime | None = None,
    status: str = "Present",
    source: str = "camera",
    db_path: Path = DB_PATH,
    csv_path: Path = ATTENDANCE_CSV,
) -> dict[str, Any]:
    initialize_database(db_path)
    timestamp = timestamp or datetime.now()
    attendance_date = timestamp.strftime("%Y-%m-%d")
    attendance_time = timestamp.strftime("%H:%M:%S")

    if not validate_student_id(student_id):
        return {
            "recorded": False,
            "message": f"Student ID {student_id} is invalid.",
        }

    student = get_student(student_id, db_path=db_path)
    if student is None:
        return {
            "recorded": False,
            "message": f"Student {student_id} is not registered in the database.",
        }

    with get_connection(db_path) as connection:
        existing = connection.execute(
            """
            SELECT id
            FROM attendance
            WHERE student_id = ? AND attendance_date = ?
            """,
            (student_id, attendance_date),
        ).fetchone()

        if existing:
            return {
                "recorded": False,
                "message": f"Attendance already recorded for {student['name']} on {attendance_date}.",
                "student_name": student["name"],
                "date": attendance_date,
                "time": attendance_time,
            }

        connection.execute(
            """
            INSERT INTO attendance (
                student_id,
                attendance_date,
                attendance_time,
                status,
                source,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                attendance_date,
                attendance_time,
                status,
                source,
                timestamp.isoformat(timespec="seconds"),
            ),
        )

    append_attendance_csv(
        csv_path=csv_path,
        attendance_date=attendance_date,
        student_id=student_id,
        student_name=student["name"],
        attendance_time=attendance_time,
        status=status,
    )

    return {
        "recorded": True,
        "message": f"Attendance marked for {student['name']} at {attendance_time}.",
        "student_name": student["name"],
        "date": attendance_date,
        "time": attendance_time,
    }


def append_attendance_csv(
    csv_path: Path,
    attendance_date: str,
    student_id: str,
    student_name: str,
    attendance_time: str,
    status: str,
) -> None:
    ensure_attendance_csv(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [attendance_date, student_id, student_name, attendance_time, status]
        )


def list_attendance(
    attendance_date: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            attendance.attendance_date AS date,
            attendance.attendance_time AS time,
            attendance.status AS status,
            attendance.source AS source,
            students.id AS student_id,
            students.name AS student_name
        FROM attendance
        INNER JOIN students ON students.id = attendance.student_id
    """
    params: tuple[Any, ...] = ()

    if attendance_date:
        query += " WHERE attendance.attendance_date = ?"
        params = (attendance_date,)

    query += " ORDER BY attendance.attendance_date DESC, attendance.attendance_time DESC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def list_student_attendance(
    student_id: str,
    limit: int | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            attendance.attendance_date AS date,
            attendance.attendance_time AS time,
            attendance.status AS status,
            attendance.source AS source,
            students.id AS student_id,
            students.name AS student_name
        FROM attendance
        INNER JOIN students ON students.id = attendance.student_id
        WHERE students.id = ?
        ORDER BY attendance.attendance_date DESC, attendance.attendance_time DESC
    """
    params: list[Any] = [student_id]

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection(db_path) as connection:
        rows = connection.execute(query, tuple(params)).fetchall()

    return [dict(row) for row in rows]


def get_total_attendance_days(db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(DISTINCT attendance_date) AS total_days FROM attendance"
        ).fetchone()

    return int(row["total_days"]) if row else 0


def count_present_days(student_id: str, db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS present_days
            FROM attendance
            WHERE student_id = ? AND status = 'Present'
            """,
            (student_id,),
        ).fetchone()

    return int(row["present_days"]) if row else 0


def count_student_attendance_days(student_id: str, db_path: Path = DB_PATH) -> int:
    student = get_student(student_id, db_path=db_path)
    if student is None:
        return 0

    created_at = str(student.get("created_at") or "")
    created_date = created_at.split("T", 1)[0] if created_at else ""
    if not created_date:
        return 0

    try:
        start_date = datetime.strptime(created_date, "%Y-%m-%d").date()
    except ValueError:
        return 0

    today = date.today()
    if start_date > today:
        return 0

    total_days = 0
    current_day = start_date
    while current_day <= today:
        if current_day.weekday() < 5:
            total_days += 1
        current_day += timedelta(days=1)

    return max(total_days, 1)


def get_student_dashboard(student_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    student = get_student(student_id, db_path=db_path)
    if student is None:
        return None

    history = list_student_attendance(student_id=student_id, limit=20, db_path=db_path)
    present_days = count_present_days(student_id=student_id, db_path=db_path)
    total_days = count_student_attendance_days(student_id=student_id, db_path=db_path)
    attendance_rate = round((present_days / total_days) * 100, 1) if total_days else 0.0
    last_marked = history[0] if history else None
    today = date.today().strftime("%Y-%m-%d")
    today_status = "Absent"

    for row in history:
        if row["date"] == today:
            today_status = row["status"]
            break

    return {
        **student,
        "present_days": present_days,
        "total_days": total_days,
        "attendance_rate": attendance_rate,
        "status": today_status,
        "last_marked": last_marked,
        "history": history,
    }


def get_daily_summary(attendance_day: date | None = None, db_path: Path = DB_PATH) -> dict[str, Any]:
    attendance_day = attendance_day or date.today()
    attendance_date = attendance_day.strftime("%Y-%m-%d")
    students = list_students(db_path=db_path)
    attendance = list_attendance(attendance_date=attendance_date, db_path=db_path)

    return {
        "date": attendance_date,
        "registered_students": len(students),
        "present_students": len(attendance),
        "absent_students": max(len(students) - len(attendance), 0),
    }
