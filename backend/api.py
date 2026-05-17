from __future__ import annotations

import base64
import os
import subprocess
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request

from database import (
    get_daily_summary,
    get_student_dashboard,
    initialize_database,
    list_attendance,
    list_students,
    validate_student_id,
)
from delete_student import delete_registered_student
from haarcascade_utils import create_face_detector, detect_faces
from main import DEFAULT_THRESHOLD, import_runtime_dependencies, load_encodings, process_frame
from register_student import (
    extract_face_image,
    is_distinct_face,
    save_student_images_from_rgb_frames,
)

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = sys.executable
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

app = Flask(__name__)
registration_lock = Lock()
recognition_lock = Lock()
recognition_detector: Any | None = None
registration_detector: Any | None = None
encodings_cache: dict[str, list[Any]] | None = None
encodings_warning: str = ""
recognition_sessions: dict[str, dict[str, Any]] = {}
registration_sessions: dict[str, dict[str, Any]] = {}
SESSION_TTL_MINUTES = 15


def json_error(message: str, status_code: int = 400, logs: str = ""):
    payload: dict[str, Any] = {"success": False, "message": message}
    if logs:
        payload["logs"] = logs
    return jsonify(payload), status_code


def run_subprocess(args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        [PYTHON_EXECUTABLE, *args],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    logs = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
    return completed.returncode, logs


def decode_image_data_url(data_url: str):
    cv2, np = import_runtime_dependencies()
    if not data_url or "," not in data_url:
        raise RuntimeError("Browser image payload is invalid.")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError("Could not decode browser image.")

    return frame


def ensure_recognition_resources(force_reload: bool = False) -> tuple[Any, dict[str, list[Any]]]:
    global recognition_detector, encodings_cache, encodings_warning

    if recognition_detector is None:
        recognition_detector = create_face_detector()

    if encodings_cache is None or force_reload:
        try:
            encodings_cache = load_encodings()
            encodings_warning = ""
        except (FileNotFoundError, ValueError) as error:
            encodings_cache = {}
            encodings_warning = str(error)
        recognition_sessions.clear()

    return recognition_detector, encodings_cache


def ensure_registration_detector() -> Any:
    global registration_detector
    if registration_detector is None:
        registration_detector = create_face_detector()
    return registration_detector


def get_session_state(session_id: str) -> dict[str, Any]:
    now = datetime.now()
    cutoff = now - timedelta(minutes=SESSION_TTL_MINUTES)

    for existing_session_id in list(recognition_sessions):
        last_seen = recognition_sessions[existing_session_id].get("last_seen")
        if isinstance(last_seen, datetime) and last_seen < cutoff:
            recognition_sessions.pop(existing_session_id, None)

    state = recognition_sessions.get(session_id)
    if state is None:
        state = {
            "recent_recognitions": {},
            "embedding_histories": {},
            "last_seen": now,
        }
        recognition_sessions[session_id] = state
    else:
        state["last_seen"] = now

    return state


def get_registration_session_state(session_id: str) -> dict[str, Any]:
    now = datetime.now()
    cutoff = now - timedelta(minutes=SESSION_TTL_MINUTES)

    for existing_session_id in list(registration_sessions):
        last_seen = registration_sessions[existing_session_id].get("last_seen")
        if isinstance(last_seen, datetime) and last_seen < cutoff:
            registration_sessions.pop(existing_session_id, None)

    state = registration_sessions.get(session_id)
    if state is None:
        state = {
            "accepted_count": 0,
            "last_saved_face_bgr": None,
            "last_seen": now,
        }
        registration_sessions[session_id] = state
    else:
        state["last_seen"] = now

    return state


def serialize_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []

    for detection in detections:
        top, right, bottom, left = detection["location"]
        color = detection.get("color", (0, 255, 0))
        serialized.append(
            {
                "label": detection.get("label", ""),
                "recognized": bool(detection.get("recognized", False)),
                "student_id": detection.get("student_id"),
                "student_name": detection.get("student_name"),
                "confidence": detection.get("confidence"),
                "attendance_recorded": bool(detection.get("attendance_recorded", False)),
                "attendance_message": detection.get("attendance_message", ""),
                "color": list(color),
                "location": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                },
            }
        )

    return serialized


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    return json_error(str(error), 500)


@app.route("/api/health", methods=["GET"])
def health():
    initialize_database()
    ensure_recognition_resources()
    return jsonify({"success": True, "message": "API is running.", "encodings_warning": encodings_warning})


@app.route("/api/summary", methods=["GET"])
def summary():
    initialize_database()
    date_value = request.args.get("date")
    attendance_day = None

    if date_value:
        try:
            attendance_day = datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError:
            return json_error("Date must be in YYYY-MM-DD format.", 400)

    return jsonify({"success": True, "summary": get_daily_summary(attendance_day=attendance_day)})


@app.route("/api/students", methods=["GET"])
def students():
    initialize_database()
    student_rows = list_students()
    enriched_students = []

    for student in student_rows:
        dashboard = get_student_dashboard(student["id"])
        if dashboard is None:
            continue

        enriched_students.append(
            {
                "id": dashboard["id"],
                "name": dashboard["name"],
                "image_dir": dashboard.get("image_dir"),
                "encoding_count": dashboard.get("encoding_count", 0),
                "attendance_rate": dashboard["attendance_rate"],
                "present_days": dashboard["present_days"],
                "total_days": dashboard["total_days"],
                "status": dashboard["status"],
                "last_marked": dashboard["last_marked"],
                "created_at": dashboard.get("created_at"),
                "updated_at": dashboard.get("updated_at"),
            }
        )

    return jsonify({"success": True, "students": enriched_students})


@app.route("/api/students/<student_id>", methods=["GET"])
def student_detail(student_id: str):
    initialize_database()
    dashboard = get_student_dashboard(student_id)
    if dashboard is None:
        return json_error(f"Student {student_id} was not found.", 404)

    return jsonify({"success": True, "student": dashboard})


@app.route("/api/attendance", methods=["GET"])
def attendance():
    initialize_database()
    date_value = request.args.get("date")
    rows = list_attendance(attendance_date=date_value)
    return jsonify({"success": True, "attendance": rows})


@app.route("/api/register-student-browser", methods=["POST"])
def register_student_browser():
    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()
    student_name = str(payload.get("student_name", "")).strip()
    images = payload.get("images", [])

    if not student_id or not student_name:
        return json_error("Student ID and student name are required.", 400)
    if not validate_student_id(student_id):
        return json_error(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long.",
            400,
        )

    if not isinstance(images, list) or not images:
        return json_error("Browser registration needs one or more captured images.", 400)

    if registration_lock.locked():
        return json_error("A registration task is already running.", 409)

    with registration_lock:
        try:
            rgb_frames = []
            cv2, _ = import_runtime_dependencies()

            for image in images:
                frame_bgr = decode_image_data_url(str(image))
                rgb_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            registration_result = save_student_images_from_rgb_frames(
                student_id=student_id,
                student_name=student_name,
                rgb_frames=rgb_frames,
            )

            if registration_result["captured"] == 0:
                return json_error(registration_result["message"], 400)

            encode_code, encode_logs = run_subprocess(
                ["encode_faces.py", "--student-id", student_id]
            )
            if encode_code != 0:
                return json_error("Encoding process failed.", 500, encode_logs)

            ensure_recognition_resources(force_reload=True)
            student = get_student_dashboard(student_id)

            return jsonify(
                {
                    "success": True,
                    "message": f"Student {student_name} registered and encoded from browser camera.",
                    "student": student,
                    "saved_images": registration_result["captured"],
                    "logs": encode_logs,
                }
            )
        except RuntimeError as error:
            return json_error(str(error), 500)


@app.route("/api/check-registration-frame", methods=["POST"])
def check_registration_frame():
    payload = request.get_json(silent=True) or {}
    image = str(payload.get("image", "")).strip()
    session_id = str(payload.get("session_id", "")).strip() or "default"

    if not image:
        return json_error("Registration frame is required.", 400)

    cv2, np = import_runtime_dependencies()
    detector = ensure_registration_detector()
    frame_bgr = decode_image_data_url(image)
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    state = get_registration_session_state(session_id)
    detections = detect_faces(detector, rgb_frame, min_size=40)
    primary_face = (
        max(
            detections,
            key=lambda item: (item.location[1] - item.location[3]) * (item.location[2] - item.location[0]),
        )
        if detections
        else None
    )
    face_image = extract_face_image(rgb_frame, primary_face, cv2) if primary_face is not None else None

    if not detections:
        return jsonify(
            {
                "success": True,
                "face_detected": False,
                "accepted": False,
                "captured": state["accepted_count"],
                "message": "Move your face inside the guide.",
            }
        )

    if face_image is None or getattr(face_image, "size", 0) == 0:
        return jsonify(
            {
                "success": True,
                "face_detected": True,
                "accepted": False,
                "captured": state["accepted_count"],
                "message": "Face detected, but hold still for a cleaner capture.",
            }
        )

    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    if not is_distinct_face(face_bgr, state["last_saved_face_bgr"], cv2, np):
        return jsonify(
            {
                "success": True,
                "face_detected": True,
                "accepted": False,
                "captured": state["accepted_count"],
                "message": "Turn slightly left or right for the next capture.",
            }
        )

    state["last_saved_face_bgr"] = face_bgr
    state["accepted_count"] += 1
    return jsonify(
        {
            "success": True,
            "face_detected": True,
            "accepted": True,
            "captured": state["accepted_count"],
            "message": "Good capture.",
        }
    )


@app.route("/api/encode-student", methods=["POST"])
def encode_student():
    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()

    if not student_id:
        return json_error("Student ID is required.", 400)
    if not validate_student_id(student_id):
        return json_error(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long.",
            400,
        )

    return_code, logs = run_subprocess(["encode_faces.py", "--student-id", student_id])
    if return_code != 0:
        return json_error("Encoding process failed.", 500, logs)

    ensure_recognition_resources(force_reload=True)
    student = get_student_dashboard(student_id)
    return jsonify(
        {
            "success": True,
            "message": f"Student {student_id} encoding updated.",
            "student": student,
            "logs": logs,
        }
    )


@app.route("/api/recognize-frame", methods=["POST"])
def recognize_frame():
    payload = request.get_json(silent=True) or {}
    image = str(payload.get("image", "")).strip()
    threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
    session_id = str(payload.get("session_id", "")).strip() or "default"

    if not image:
        return json_error("Attendance frame is required.", 400)

    detector, encodings_by_id = ensure_recognition_resources()
    frame_bgr = decode_image_data_url(image)

    with recognition_lock:
        session_state = get_session_state(session_id)
        detections = process_frame(
            frame=frame_bgr,
            detector=detector,
            encodings_by_id=encodings_by_id,
            threshold=threshold,
            recent_recognitions=session_state["recent_recognitions"],
            embedding_histories=session_state["embedding_histories"],
        )

    return jsonify(
        {
            "success": True,
            "detections": serialize_detections(detections),
            "summary": get_daily_summary(attendance_day=datetime.now().date()),
            "encodings_warning": encodings_warning,
        }
    )


@app.route("/api/students/<student_id>", methods=["DELETE"])
def delete_student_api(student_id: str):
    if not validate_student_id(student_id):
        return json_error(
            "Student ID must contain only letters, numbers, underscores, or hyphens and be at most 50 characters long.",
            400,
        )

    try:
        delete_registered_student(student_id)
    except Exception as error:
        return json_error(str(error), 500)

    ensure_recognition_resources(force_reload=True)
    return jsonify(
        {
            "success": True,
            "message": f"Student {student_id} deleted.",
        }
    )


import os

if __name__ == "__main__":
    initialize_database()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )