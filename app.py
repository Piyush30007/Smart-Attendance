from __future__ import annotations

import importlib
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import database as database_module
import delete_student as delete_student_module
import encode_faces as encode_faces_module
import main as main_module
import register_student as register_student_module

database_module = importlib.reload(database_module)
delete_student_module = importlib.reload(delete_student_module)
encode_faces_module = importlib.reload(encode_faces_module)
main_module = importlib.reload(main_module)
register_student_module = importlib.reload(register_student_module)

get_daily_summary = database_module.get_daily_summary
list_attendance = database_module.list_attendance
list_students = database_module.list_students
delete_registered_student = delete_student_module.delete_registered_student
generate_face_encodings = encode_faces_module.generate_face_encodings
run_attendance = main_module.run_attendance
capture_student_images = register_student_module.capture_student_images


def capture_logs(func, *args, **kwargs):
    output = io.StringIO()
    result = None
    error_message = None

    try:
        with redirect_stdout(output), redirect_stderr(output):
            result = func(*args, **kwargs)
    except Exception as error:
        traceback.print_exc(file=output)
        error_message = str(error)

    return result, output.getvalue(), error_message


def render_style() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: #f8fafc;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1rem 1.2rem;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            background: #f8fbff;
        }
        .small-note {
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_logs(title: str, logs: str) -> None:
    if logs.strip():
        with st.expander(title, expanded=True):
            st.code(logs, language="text")


def registration_section() -> None:
    st.subheader("Register Student")
    st.markdown(
        '<div class="card"><span class="small-note">This captures from the live camera, saves images, and then encodes only that student immediately.</span></div>',
        unsafe_allow_html=True,
    )

    with st.form("register_form"):
        col1, col2 = st.columns(2)
        student_id = col1.text_input("Student ID")
        student_name = col2.text_input("Student Name")

        col3, col4, col5 = st.columns(3)
        image_count = col3.number_input("Images", min_value=5, max_value=100, value=20, step=5)
        camera_index = col4.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
        manual_mode = col5.checkbox("Manual Capture", value=False)

        col6, col7 = st.columns(2)
        sample_every = col6.number_input("Sample Every N Frames", min_value=1, max_value=20, value=5, step=1)
        average_window = col7.number_input("Average Window", min_value=1, max_value=10, value=4, step=1)

        submitted = st.form_submit_button("Register And Encode", use_container_width=True)

    if submitted:
        if not student_id.strip() or not student_name.strip():
            st.error("Student ID and Student Name are required.")
            return

        st.info("Camera window will open for registration. Close it or let capture finish to continue encoding.")
        captured_path, registration_logs, registration_error = capture_logs(
            capture_student_images,
            student_id=student_id.strip(),
            student_name=student_name.strip(),
            image_count=int(image_count),
            camera_index=int(camera_index),
            sample_every_frames=int(sample_every),
            averaging_window=int(average_window),
            manual_mode=manual_mode,
        )
        show_logs("Registration Logs", registration_logs)

        if registration_error:
            st.error(f"Registration failed: {registration_error}")
            return

        if not captured_path:
            st.warning("Registration did not complete, so encoding was skipped.")
            return

        _, encoding_logs, encoding_error = capture_logs(
            generate_face_encodings,
            student_id=student_id.strip(),
        )
        show_logs("Encoding Logs", encoding_logs)

        if encoding_error:
            st.error(f"Encoding failed: {encoding_error}")
            return

        st.success(f"Student {student_name.strip()} ({student_id.strip()}) was registered and encoded.")


def attendance_section() -> None:
    st.subheader("Attendance")
    st.markdown(
        '<div class="card"><span class="small-note">This starts the live attendance camera session. The Streamlit page will wait until you close the OpenCV window.</span></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    camera_index = col1.number_input("Attendance Camera Index", min_value=0, max_value=10, value=0, step=1, key="attendance_camera_index")
    threshold = col2.number_input("Match Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="attendance_threshold")
    frame_skip = col3.number_input("Frame Skip", min_value=1, max_value=10, value=2, step=1, key="attendance_frame_skip")

    if st.button("Start Attendance Camera", use_container_width=True):
        st.info("Attendance camera started. Close the OpenCV window with 'q' when done.")
        _, attendance_logs, attendance_error = capture_logs(
            run_attendance,
            camera_index=int(camera_index),
            threshold=float(threshold),
            process_every_n_frames=int(frame_skip),
        )
        show_logs("Attendance Logs", attendance_logs)

        if attendance_error:
            st.error(f"Attendance failed: {attendance_error}")


def delete_section() -> None:
    st.subheader("Delete Student")
    student_id_to_delete = st.text_input("Student ID To Delete")
    if st.button("Delete Student", use_container_width=True):
        if not student_id_to_delete.strip():
            st.error("Enter a student ID first.")
            return
        _, delete_logs, delete_error = capture_logs(delete_registered_student, student_id_to_delete.strip())
        show_logs("Delete Logs", delete_logs)

        if delete_error:
            st.error(f"Delete failed: {delete_error}")
            return

        st.success(f"Delete flow completed for student ID {student_id_to_delete.strip()}.")


def dashboard_section() -> None:
    st.subheader("Dashboard")
    selected_date = st.date_input("Attendance Date", value=date.today(), key="dashboard_date")
    summary = get_daily_summary(attendance_day=selected_date)
    attendance_rows = list_attendance(attendance_date=selected_date.strftime("%Y-%m-%d"))
    students = list_students()

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Students", summary["registered_students"])
    col2.metric("Present Today", summary["present_students"])
    col3.metric("Absent Today", summary["absent_students"])

    st.markdown("### Students")
    if students:
        st.dataframe(students, use_container_width=True)
    else:
        st.info("No registered students yet.")

    st.markdown("### Attendance")
    if attendance_rows:
        st.dataframe(attendance_rows, use_container_width=True)
    else:
        st.info("No attendance records for the selected date.")


def main() -> None:
    st.set_page_config(
        page_title="Smart Attendance Frontend",
        page_icon="SA",
        layout="wide",
    )
    render_style()
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Smart Attendance Frontend</h2>
            <p style="margin:0.35rem 0 0 0;">Register students, encode them immediately, launch attendance, and review records from one screen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Register Student", "Attendance", "Dashboard", "Delete Student"]
    )

    with tab1:
        registration_section()
    with tab2:
        attendance_section()
    with tab3:
        dashboard_section()
    with tab4:
        delete_section()


if __name__ == "__main__":
    main()
