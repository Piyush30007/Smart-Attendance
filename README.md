# Smart Attendance System

This project implements the pipeline:

`Camera -> Face Detection -> Face Encoding -> Database Matching -> Attendance Log`

## Tech Stack

- Python
- OpenCV
- MTCNN
-HaarCascade 
- face_recognition
- SQLite


## Project Structure

```text
smart-attendance/
|-- dataset/              # stored face images
|-- encodings/            # saved face embeddings
|-- attendance.csv        # CSV attendance log
|-- smart_attendance.db   # SQLite database (auto-created)
|-- register_student.py   # manual student registration from camera
|-- delete_student.py     # delete a registered student and cleanup files
|-- main.py               # run recognition and attendance marking
|-- encode_faces.py       # create embeddings from dataset images
|-- database.py           # database helpers
`-- app.py                # optional Streamlit dashboard
```

## Manual Registration

```bash
python register_student.py
```

This will ask for:

- student ID
- student name
- number of images to capture
- camera index

By default, registration is automatic:

- it reads live camera video
- uses Haar cascade face detection on each live frame
- samples face crops every few frames
- averages several samples into one saved image for a cleaner dataset
- aligns faces using Haar Cascade eye keypoints before saving

Controls during registration:

- `q` stops registration early

If you want the old manual mode:

```bash
python register_student.py --manual
```

You can also pass values directly:

```bash
python register_student.py --student-id 101 --student-name "John Doe" --images 20
```

Default capture count is now `40` images per student. You can still change it with `--images`.

Useful tuning flags:

- `--sample-every 5`
- `--average-window 4`

## Dataset Format

Create one folder per student inside `dataset/`.

Recommended folder naming:

- `101__John_Doe`
- `102__Jane_Smith`

Each student folder should contain several face images of the same person.

## Setup

For manual registration only:

```bash
python -m pip install -r requirements-registration.txt
```

For the full face-recognition pipeline:

```bash
python -m pip install -r requirements.txt
```

## Python Version Note

If you are using Python `3.14`, `opencv-python` should install, but `face_recognition` often fails on Windows because it depends on `dlib`.

As of March 26, 2026:

- `face_recognition` on PyPI is still version `1.3.0` from February 20, 2020
- its PyPI page says Windows is not officially supported
- its listed Python classifiers only go up to `Python 3.8`

For the smoothest full setup on Windows, use Python `3.10` or `3.11` in a virtual environment.

Also keep `numpy` on the `1.x` line for this stack. If you installed NumPy `2.x`, fix it with:

```bash
python -m pip install "numpy<2"
```

For this codebase, use the MTCNN package line:

```bash
python -m pip install "mtcnn[tensorflow]"
```

## 1. Generate Encodings

```bash
python encode_faces.py
```

This will:

- read every student folder in `dataset/`
- use MTCNN/HaarCascade  to detect the face region in each image
- align each face using MTCNN/Haar Cascade eye keypoints
- generate face embeddings
- save embeddings to `encodings/face_encodings.pkl`
- register or update student records in SQLite

For production-style incremental updates, encode only one student:

```bash
python encode_faces.py --student-id 207
```

To force a full rebuild from all dataset folders:

```bash
python encode_faces.py --rebuild-all
```

## 2. Run Smart Attendance

```bash
python main.py
```

This will:

- open the camera
- detect faces with MTCNN/HaarCascade 
- align the detected face before encoding
- encode faces in real time
- match them against saved embeddings
- fetch student details from SQLite
- write attendance to both SQLite and `attendance.csv`

Quit with `q`.

## 3. View Attendance in Terminal

```bash
python main.py --view
```

For a specific date:

```bash
python main.py --view --date 2026-03-25
```

## 4. Optional Web Dashboard

```bash
streamlit run app.py
```

## Delete A Registered Student

```bash
python delete_student.py --student-id 101
```

This removes the student from:

- SQLite
- dataset images
- saved encodings
- `attendance.csv`

## Notes

- One attendance record is stored per student per day.
- The database is the source of truth for registered students and attendance records.
- The CSV file is kept as an export-friendly attendance log.
- Registration can be done manually with `register_student.py` before encoding faces.
