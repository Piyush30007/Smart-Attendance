import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import axios from 'axios'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const API = API_BASE ? `${API_BASE}/api` : '/api'
const TODAY = localDate()
const POLL_MS = 6000
const REG_TOTAL_FRAMES = 12
const REG_FRAME_DELAY = 550
const ATTEND_INTERVAL = 650
const WEB_DETECTOR_MODEL = 'haar'

// ─── App ─────────────────────────────────────────────────────────────────────

function App() {
  const [view, setView] = useState('admin')
  const [date, setDate] = useState(TODAY)
  const [summary, setSummary] = useState({ registered_students: 0, present_students: 0, absent_students: 0 })
  const [students, setStudents] = useState([])
  const [attendanceRows, setAttendanceRows] = useState([])
  const [selectedStudentId, setSelectedStudentId] = useState('')
  const [selectedStudent, setSelectedStudent] = useState(null)
  const [statusMsg, setStatusMsg] = useState('Ready.')
  const [error, setError] = useState('')

  const refresh = useCallback(async (targetDate) => {
    try {
      const [summaryRes, studentsRes, attendanceRes] = await Promise.all([
        axios.get(`${API}/summary`, { params: { date: targetDate } }),
        axios.get(`${API}/students`),
        axios.get(`${API}/attendance`, { params: { date: targetDate } }),
      ])
      const nextStudents = studentsRes.data.students ?? []
      setSummary(summaryRes.data.summary ?? {})
      setStudents(nextStudents)
      setAttendanceRows(attendanceRes.data.attendance ?? [])
      setError('')
      setSelectedStudentId(prev => prev || nextStudents[0]?.id || '')
    } catch (err) {
      setError(extractError(err, 'Could not load backend data.'))
    }
  }, [])

  const fetchStudent = useCallback(async (studentId) => {
    if (!studentId) return
    try {
      const res = await axios.get(`${API}/students/${studentId}`)
      setSelectedStudent(res.data.student)
    } catch (err) {
      setError(extractError(err, `Could not load student ${studentId}.`))
    }
  }, [])

  useEffect(() => {
    queueMicrotask(() => { void refresh(date) })
    const id = setInterval(() => refresh(date), POLL_MS)
    return () => clearInterval(id)
  }, [date, refresh])

  useEffect(() => {
    if (selectedStudentId) {
      queueMicrotask(() => { void fetchStudent(selectedStudentId) })
    }
  }, [selectedStudentId, fetchStudent])

  const handleDeleteStudent = useCallback(async (studentId) => {
    setError('')
    try {
      const res = await axios.delete(`${API}/students/${studentId}`)
      setStatusMsg(res.data.message)
      await refresh(date)
      setSelectedStudentId(prev => prev === studentId ? '' : prev)
    } catch (err) {
      setError(extractError(err, `Delete failed for ${studentId}.`))
    }
  }, [date, refresh])

  return (
    <main className="shell">
      <header className="hero-panel">
        <div className="hero-left">
          <span className="eyebrow">Smart Attendance</span>
          <h1>Face Recognition <br /><em>Dashboard</em></h1>
          <p className="hero-copy">Live browser-based attendance system</p>
        </div>
        <div className="hero-stats">
          <StatBadge label="Students" value={summary.registered_students} />
          <StatBadge label="Present" value={summary.present_students} accent />
          <StatBadge label="Absent" value={summary.absent_students} />
        </div>
      </header>

      <nav className="toolbar">
        <div className="switcher">
          <button className={view === 'admin' ? 'switch active' : 'switch'} onClick={() => setView('admin')}>Admin</button>
          <button className={view === 'student' ? 'switch active' : 'switch'} onClick={() => setView('student')}>Students</button>
        </div>
        <div className="toolbar-right">
          <div className="status-chip">
            <span className="live-dot" />
            <span>{statusMsg}</span>
          </div>
          <label className="date-field">
            <span>Date</span>
            <input type="date" value={date} onChange={e => setDate(e.target.value)} />
          </label>
        </div>
      </nav>

      {error && <div className="error-banner">{error}</div>}

      {view === 'student' ? (
        <StudentView
          students={students}
          selectedStudentId={selectedStudentId}
          student={selectedStudent}
          onSelect={setSelectedStudentId}
        />
      ) : (
        <AdminView
          summary={summary}
          students={students}
          attendanceRows={attendanceRows}
          date={date}
          setStatusMsg={setStatusMsg}
          setError={setError}
          refresh={refresh}
          fetchStudent={fetchStudent}
          selectedStudentId={selectedStudentId}
          onSelectStudent={id => { setSelectedStudentId(id); setView('student') }}
          onDeleteStudent={handleDeleteStudent}
        />
      )}
    </main>
  )
}

// ─── StudentView ──────────────────────────────────────────────────────────────

function StudentView({ students, selectedStudentId, student, onSelect }) {
  return (
    <section className="dashboard-grid">
      <div className="panel chip-row-panel">
        {students.length === 0 && <p className="muted">No students registered yet.</p>}
        {students.map(s => (
          <button
            key={s.id}
            className={s.id === selectedStudentId ? 'chip active' : 'chip'}
            onClick={() => onSelect(s.id)}
          >
            <span className="chip-avatar">{s.name.charAt(0)}</span>
            <span>{s.name}</span>
            <span className="chip-id">{s.id}</span>
          </button>
        ))}
      </div>

      {student ? (
        <>
          <div className="panel profile-card">
            <div className="avatar">{student.name?.charAt(0)}</div>
            <div>
              <p className="eyebrow">Student Profile</p>
              <h2>{student.name}</h2>
              <p className="muted">ID: {student.id} · {student.encoding_count ?? 0} encodings</p>
            </div>
            <div className="profile-meta">
              <InfoPair label="Today" value={student.status || 'Unknown'} highlight={student.status === 'Present'} />
              <InfoPair label="Last Marked" value={formatLastMarked(student.last_marked)} />
              <InfoPair label="Present Days" value={student.present_days ?? 0} />
              <InfoPair label="Total Days" value={student.total_days ?? 0} />
              <InfoPair label="Attendance Rate" value={formatRate(student.attendance_rate)} highlight />
              <InfoPair label="Updated" value={formatDateTime(student.updated_at)} />
            </div>
          </div>

          <div className="panel history-panel">
            <h3>Attendance History</h3>
            <AttendanceTable rows={student.history ?? []} compact />
          </div>
        </>
      ) : (
        <div className="panel empty-state">
          <div className="empty-icon">👤</div>
          <h3>Select a student above</h3>
          <p>Click any student chip to view their profile.</p>
        </div>
      )}
    </section>
  )
}

// ─── AdminView ────────────────────────────────────────────────────────────────

function AdminView({ students, attendanceRows, date, setStatusMsg, setError, refresh, fetchStudent, selectedStudentId, onSelectStudent, onDeleteStudent }) {
  return (
    <section className="dashboard-grid admin-grid">
      <AttendancePanel
        date={date}
        setStatusMsg={setStatusMsg}
        setError={setError}
        refresh={refresh}
        fetchStudent={fetchStudent}
        selectedStudentId={selectedStudentId}
      />

      <RegisterPanel
        date={date}
        setStatusMsg={setStatusMsg}
        setError={setError}
        refresh={refresh}
        fetchStudent={fetchStudent}
        onSelectStudent={onSelectStudent}
      />

      <div className="panel panel-wide">
        <div className="section-head">
          <h3>Registered Students</h3>
          <span className="badge">{students.length} total</span>
        </div>
        <StudentsTable
          students={students}
          date={date}
          setStatusMsg={setStatusMsg}
          setError={setError}
          refresh={refresh}
          fetchStudent={fetchStudent}
          selectedStudentId={selectedStudentId}
          onSelectStudent={onSelectStudent}
          onDeleteStudent={onDeleteStudent}
        />
      </div>

      <div className="panel panel-wide">
        <div className="section-head">
          <h3>Today's Attendance</h3>
          <span className="badge">{attendanceRows.length} marked</span>
        </div>
        <AttendanceTable rows={attendanceRows} />
      </div>
    </section>
  )
}

// ─── AttendancePanel (Live Recognition) ──────────────────────────────────────

function AttendancePanel({ date, setStatusMsg, setError, refresh, fetchStudent, selectedStudentId }) {
  const [threshold, setThreshold] = useState(0.55)
  const [intervalMs, setIntervalMs] = useState(ATTEND_INTERVAL)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(false)
  const [stream, setStream] = useState(null)
  const [detections, setDetections] = useState([])

  const videoRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const overlayRef = useRef(null)
  const inFlightRef = useRef(false)
  const timerRef = useRef(null)
  const sessionIdRef = useRef(`s-${Date.now()}`)
  const thresholdRef = useRef(threshold)
  thresholdRef.current = threshold
  const intervalRef = useRef(intervalMs)
  intervalRef.current = intervalMs

  // Resize observer for accurate overlay scaling
  const videoDimsRef = useRef({ width: 0, height: 0 })
  useEffect(() => {
    const video = videoRef.current
    if (!video || !open) return
    const ro = new ResizeObserver(() => {
      const r = video.getBoundingClientRect()
      videoDimsRef.current = { width: r.width, height: r.height }
    })
    ro.observe(video)
    return () => ro.disconnect()
  }, [open, stream])

  // Open camera when modal opens
  useEffect(() => {
    if (!open) return
    let cancelled = false
    ;(async () => {
      try {
        const s = await openCamera(videoRef.current)
        if (cancelled) { stopStream(s); return }
        setStream(s)
        setActive(true)
        setError('')
      } catch (err) {
        if (!cancelled) { setError(extractError(err, 'Camera permission denied.')); setOpen(false) }
      }
    })()
    return () => { cancelled = true }
  }, [open, setError])

  // Recognition loop
  useEffect(() => {
    const overlayCanvas = overlayRef.current
    if (!active || !stream) { clearCanvas(overlayCanvas); return }
    let cancelled = false

    const loop = async () => {
      if (cancelled) return
      if (!inFlightRef.current) {
        const img = captureFrame(videoRef.current, captureCanvasRef.current, 640)
        if (img) {
          inFlightRef.current = true
          try {
            const res = await axios.post(`${API}/recognize-frame`, {
              image: img,
              threshold: thresholdRef.current,
              session_id: sessionIdRef.current,
              detector_model: WEB_DETECTOR_MODEL,
            })
            const dets = res.data.detections ?? []
            setDetections(dets)
            // Draw boxes with CORRECT coordinate mapping
            drawDetectionBoxes(overlayCanvas, videoRef.current, dets, videoDimsRef.current, 640)
            const names = dets.filter(d => d.recognized).map(d => d.student_name || d.label)
            setStatusMsg(names.length ? `✓ Recognized: ${names.join(', ')}` : dets.length ? 'Face detected — not recognized' : 'Scanning…')
            if (dets.some(d => d.attendance_recorded)) {
              await refresh(date)
              if (selectedStudentId) await fetchStudent(selectedStudentId)
            }
          } catch (err) {
            setError(extractError(err, 'Recognition error.'))
          } finally {
            inFlightRef.current = false
          }
        }
      }
      if (!cancelled) timerRef.current = setTimeout(loop, intervalRef.current)
    }

    timerRef.current = setTimeout(loop, 300)
    return () => {
      cancelled = true
      clearTimeout(timerRef.current)
      clearCanvas(overlayCanvas)
    }
  }, [active, stream, date, refresh, fetchStudent, selectedStudentId, setError, setStatusMsg])

  const close = () => {
    setActive(false)
    setOpen(false)
    setDetections([])
    stopStream(stream)
    setStream(null)
    clearCanvas(overlayRef.current)
  }

  useEffect(() => () => stopStream(stream), [stream])

  return (
    <div className="panel attend-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Live Recognition</p>
          <h3>Browser Attendance Camera</h3>
        </div>
        <div className="attend-dot-wrap">
          <span className={active ? 'status-dot active' : 'status-dot'} />
          <span>{active ? 'Live' : 'Offline'}</span>
        </div>
      </div>

      <div className="form-row">
        <label className="field">
          <span>Match Threshold</span>
          <input type="number" min="0.1" max="1" step="0.05" value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>Interval (ms)</span>
          <input type="number" min="300" step="100" value={intervalMs} onChange={e => setIntervalMs(Number(e.target.value))} />
        </label>
      </div>

      <div className="btn-row">
        <button className="btn-primary" disabled={active} onClick={() => setOpen(true)}>Start Attendance</button>
        <button className="btn-ghost" disabled={!active} onClick={close}>Stop</button>
      </div>

      {open && createPortal(
        <div className="modal-backdrop" onClick={close}>
          <div className="modal-card attend-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <p className="eyebrow">Attendance Camera</p>
                <h3>Face detected = attendance marked</h3>
              </div>
              <button className="btn-ghost btn-sm" onClick={close}>✕ Close</button>
            </div>

            <div className="video-stage">
              <video ref={videoRef} autoPlay muted playsInline className="video-feed" />
              <canvas ref={overlayRef} className="video-overlay" />
            </div>
            <canvas ref={captureCanvasRef} className="hidden-canvas" />

            <div className="detection-feed">
              {detections.length === 0
                ? <p className="muted center">Waiting for faces…</p>
                : detections.map((d, i) => (
                  <div key={i} className={`detection-card ${d.recognized ? 'recognized' : 'unknown'}`}>
                    <div className="detection-icon">{d.recognized ? '✓' : '?'}</div>
                    <div>
                      <strong>{d.label}</strong>
                      {d.attendance_message && <p className="det-msg">{d.attendance_message}</p>}
                    </div>
                  </div>
                ))
              }
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}

// ─── RegisterPanel (Phone-style face lock UX) ─────────────────────────────────

function RegisterPanel({ date, setStatusMsg, setError, refresh, fetchStudent, onSelectStudent }) {
  const [form, setForm] = useState({ student_id: '', student_name: '' })
  const [open, setOpen] = useState(false)
  const [stream, setStream] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | scanning | uploading | done
  const [captured, setCaptured] = useState(0)
  const [hasFace, setHasFace] = useState(false)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const capturedImagesRef = useRef([])
  const abortRef = useRef(false)
  const sessionIdRef = useRef(`reg-${Date.now()}-${Math.random().toString(36).slice(2)}`)
  const [captureMessage, setCaptureMessage] = useState('Ensure good lighting and keep your face inside the guide.')

  useEffect(() => () => stopStream(stream), [stream])

  // Open camera + start face detection preview
  useEffect(() => {
    if (!open) return
    let cancelled = false
    ;(async () => {
      try {
        const s = await openCamera(videoRef.current)
        if (cancelled) { stopStream(s); return }
        setStream(s)
        setError('')
      } catch (err) {
        if (!cancelled) { setError(extractError(err, 'Camera permission denied.')); setOpen(false) }
      }
    })()
    return () => { cancelled = true }
  }, [open, setError])

  const startCapture = async () => {
    if (!form.student_id.trim() || !form.student_name.trim()) {
      setError('Enter Student ID and Name first.')
      return
    }
    if (!stream) { setError('Camera not ready.'); return }

    setPhase('scanning')
    setCaptured(0)
    setHasFace(false)
    setCaptureMessage('Move your face into the guide and hold still.')
    capturedImagesRef.current = []
    abortRef.current = false
    sessionIdRef.current = `reg-${Date.now()}-${Math.random().toString(36).slice(2)}`
  }

  // Auto-capture when backend validates the frame as a real accepted face capture.
  useEffect(() => {
    if (phase !== 'scanning' || !stream) return
    let cancelled = false
    let timerId = null

    const loop = async () => {
      if (cancelled || abortRef.current) return
      const img = captureFrame(videoRef.current, canvasRef.current, 640)
      if (!img) {
        timerId = setTimeout(loop, REG_FRAME_DELAY)
        return
      }

      try {
        const res = await axios.post(`${API}/check-registration-frame`, {
          image: img,
          session_id: sessionIdRef.current,
          detector_model: WEB_DETECTOR_MODEL,
        })
        const info = res.data ?? {}
        setHasFace(Boolean(info.face_detected))
        setCaptureMessage(info.message || 'Hold still for the next capture.')

        if (info.accepted) {
          capturedImagesRef.current.push(img)
          const count = Number(info.captured ?? capturedImagesRef.current.length)
          setCaptured(count)
          if (count >= REG_TOTAL_FRAMES) {
            await uploadRegistration()
            return
          }
        }
      } catch (err) {
        setError(extractError(err, 'Face validation failed.'))
        setCaptureMessage('Face validation failed. Try again.')
      }

      if (!cancelled) timerId = setTimeout(loop, REG_FRAME_DELAY)
    }

    timerId = setTimeout(loop, 250)
    return () => {
      cancelled = true
      if (timerId) clearTimeout(timerId)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, stream])

  const uploadRegistration = async () => {
    setPhase('uploading')
    try {
      const res = await axios.post(`${API}/register-student-browser`, {
        student_id: form.student_id.trim(),
        student_name: form.student_name.trim(),
        images: capturedImagesRef.current,
      })
      setStatusMsg(res.data.message)
      setPhase('done')
      await refresh(date)
      if (res.data.student?.id) {
        await fetchStudent(res.data.student.id)
        onSelectStudent(res.data.student.id)
      }
      setTimeout(() => closeModal(), 2000)
    } catch (err) {
      setError(extractError(err, 'Registration failed.'))
      setPhase('scanning')
    }
  }

  const closeModal = () => {
    abortRef.current = true
    stopStream(stream)
    setStream(null)
    setOpen(false)
    setPhase('idle')
    setCaptured(0)
    setHasFace(false)
    setCaptureMessage('Ensure good lighting and keep your face inside the guide.')
    capturedImagesRef.current = []
  }

  const progress = Math.min(captured / REG_TOTAL_FRAMES, 1)
  const ellipseA = 46
  const ellipseB = 54
  const circumference = Math.PI * (3 * (ellipseA + ellipseB) - Math.sqrt((3 * ellipseA + ellipseB) * (ellipseA + 3 * ellipseB)))
  const dashOffset = circumference * (1 - progress)

  return (
    <div className="panel reg-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Registration</p>
          <h3>Register New Student</h3>
        </div>
      </div>

      <div className="form-col">
        <input
          className="field-input"
          placeholder="Student ID (e.g. S101)"
          value={form.student_id}
          onChange={e => setForm(s => ({ ...s, student_id: e.target.value }))}
        />
        <input
          className="field-input"
          placeholder="Full Name"
          value={form.student_name}
          onChange={e => setForm(s => ({ ...s, student_name: e.target.value }))}
        />
        <button className="btn-primary" onClick={() => setOpen(true)}>
          Open Camera & Register
        </button>
      </div>

      {open && createPortal(
        <div className="modal-backdrop" onClick={closeModal}>
          <div className="modal-card reg-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <p className="eyebrow">Face Registration</p>
                <h3>{form.student_name || 'Student'}</h3>
              </div>
              {phase !== 'uploading' && (
                <button className="btn-ghost btn-sm" onClick={closeModal}>✕</button>
              )}
            </div>

            {/* Phone-style face lock UI */}
            <div className="reg-stage">
              <video ref={videoRef} autoPlay muted playsInline className="reg-video" />
              <canvas ref={canvasRef} className="hidden-canvas" />

              {/* Oval face guide with circular progress ring */}
              <div className="face-guide-wrap">
                <svg className="progress-ring" viewBox="0 0 120 120">
                  {/* Background track */}
                  <ellipse cx="60" cy="60" rx="46" ry="54"
                    fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="3" />
                  {/* Progress arc */}
                  <ellipse
                    cx="60" cy="60" rx="46" ry="54"
                    fill="none"
                    stroke={phase === 'done' ? '#4ade80' : hasFace ? '#16a34a' : 'rgba(255,255,255,0.4)'}
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    strokeDasharray={`${circumference}`}
                    strokeDashoffset={`${dashOffset}`}
                    style={{ transition: 'stroke-dashoffset 0.15s ease, stroke 0.3s ease', transform: 'rotate(-90deg)', transformOrigin: '60px 60px' }}
                  />
                </svg>
                {/* Corner brackets */}
                <div className={`face-brackets ${hasFace ? 'face-detected' : ''}`}>
                  <span className="bracket tl" />
                  <span className="bracket tr" />
                  <span className="bracket bl" />
                  <span className="bracket br" />
                </div>
              </div>

              {/* Status overlay */}
              <div className="reg-status-overlay">
                {phase === 'idle' && (
                  <div className="reg-hint">
                    <p>Position your face in the oval</p>
                    <button className="btn-primary btn-sm" onClick={startCapture}>Start</button>
                  </div>
                )}
                {phase === 'scanning' && (
                  <div className="reg-hint">
                    {!hasFace
                      ? <p className="scanning-msg no-face">{captureMessage}</p>
                      : <p className="scanning-msg has-face">{captureMessage} {captured}/{REG_TOTAL_FRAMES}</p>
                    }
                    <div className="reg-counter">{Math.round(progress * 100)}%</div>
                  </div>
                )}
                {phase === 'uploading' && (
                  <div className="reg-hint uploading">
                    <div className="spinner" />
                    <p>Processing & encoding…</p>
                  </div>
                )}
                {phase === 'done' && (
                  <div className="reg-hint done">
                    <div className="done-check">✓</div>
                    <p>Registered successfully!</p>
                  </div>
                )}
              </div>
            </div>

            <p className="reg-tip">
              {phase === 'scanning'
                ? 'Turn slightly between accepted captures so the backend can collect varied views.'
                : phase === 'idle'
                ? 'Ensure good lighting and look directly at the camera'
                : phase === 'uploading'
                ? 'Please wait while faces are being encoded…'
                : 'Registration complete. Closing…'}
            </p>
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}

// ─── StudentsTable ────────────────────────────────────────────────────────────

function StudentsTable({ students, date, setStatusMsg, setError, refresh, fetchStudent, selectedStudentId, onSelectStudent, onDeleteStudent }) {
  const [busy, setBusy] = useState('')
  const [confirmId, setConfirmId] = useState('')

  const encode = async (id) => {
    setBusy(`enc:${id}`)
    setError('')
    try {
      const res = await axios.post(`${API}/encode-student`, { student_id: id })
      setStatusMsg(res.data.message)
      await refresh(date)
      if (selectedStudentId === id) await fetchStudent(id)
    } catch (err) {
      setError(extractError(err, `Encoding failed for ${id}.`))
    } finally { setBusy('') }
  }

  const del = async (id) => {
    setBusy(`del:${id}`)
    setConfirmId('')
    await onDeleteStudent(id)
    setBusy('')
  }

  if (!students.length) return <p className="muted">No students yet.</p>

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Name</th><th>Encodings</th>
            <th>Rate</th><th>Status</th><th>Last Seen</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {students.map(s => (
            <tr key={s.id}>
              <td className="mono">{s.id}</td>
              <td><strong>{s.name}</strong></td>
              <td>{s.encoding_count ?? 0}</td>
              <td>{formatRate(s.attendance_rate)}</td>
              <td><span className={s.status === 'Present' ? 'pill present' : 'pill absent'}>{s.status || '—'}</span></td>
              <td>{formatLastMarked(s.last_marked)}</td>
              <td>
                <div className="action-row">
                  <button className="btn-xs" onClick={() => onSelectStudent(s.id)}>View</button>
                  <button className="btn-xs" disabled={busy === `enc:${s.id}`} onClick={() => encode(s.id)}>
                    {busy === `enc:${s.id}` ? '…' : 'Encode'}
                  </button>
                  {confirmId === s.id ? (
                    <>
                      <button className="btn-xs danger" onClick={() => del(s.id)}>Confirm</button>
                      <button className="btn-xs" onClick={() => setConfirmId('')}>Cancel</button>
                    </>
                  ) : (
                    <button className="btn-xs danger" onClick={() => setConfirmId(s.id)}>Delete</button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Shared ───────────────────────────────────────────────────────────────────

function AttendanceTable({ rows, compact }) {
  if (!rows.length) return <p className="muted center">No records.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            {!compact && <th>Student ID</th>}
            <th>Name</th>
            <th>Time</th>
            <th>Status</th>
            {!compact && <th>Source</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.student_id}-${r.date}-${i}`}>
              <td>{r.date}</td>
              {!compact && <td className="mono">{r.student_id}</td>}
              <td>{r.student_name}</td>
              <td>{r.time}</td>
              <td><span className={r.status === 'Present' ? 'pill present' : 'pill absent'}>{r.status}</span></td>
              {!compact && <td>{r.source ?? 'camera'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function StatBadge({ label, value, accent }) {
  return (
    <div className={accent ? 'stat-badge accent' : 'stat-badge'}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  )
}

function MetricCard({ label, value, accent }) {
  return <div className={accent ? 'metric-card accent' : 'metric-card'}><span>{label}</span><strong>{value}</strong></div>
}

function InfoPair({ label, value, highlight }) {
  return (
    <div className="info-pair">
      <span>{label}</span>
      <strong className={highlight ? 'highlight' : ''}>{value}</strong>
    </div>
  )
}

// ─── Camera utilities ─────────────────────────────────────────────────────────

async function openCamera(video) {
  if (!video) throw new Error('Video element not ready.')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  })
  video.srcObject = stream
  await video.play()
  return stream
}

function stopStream(stream) {
  stream?.getTracks().forEach(t => t.stop())
}

/**
 * Capture a frame from the video element into a JPEG data URL.
 * The frame is always captured at `captureW` pixels wide (default 640).
 */
function captureFrame(video, canvas, captureW = 640) {
  if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return null
  const ratio = video.videoHeight / video.videoWidth
  canvas.width = captureW
  canvas.height = Math.round(captureW * ratio)
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/jpeg', 0.88)
}

/**
 * Draw detection boxes on the overlay canvas.
 *
 * The backend receives a frame at captureW x (captureW * videoAspect).
 * The backend then internally does:  small = resize(frame, 0.25x) → detects → *4 → returns coords in ORIGINAL frame space.
 *
 * So the detection coords are in the original capture space (captureW × captureH).
 * We need to map those to the display space (displayW × displayH).
 */
function drawDetectionBoxes(canvas, video, detections, dims, captureW) {
  if (!canvas || !video) return

  const displayW = dims?.width || video.getBoundingClientRect().width
  const displayH = dims?.height || video.getBoundingClientRect().height

  if (canvas.width !== Math.round(displayW)) canvas.width = Math.round(displayW)
  if (canvas.height !== Math.round(displayH)) canvas.height = Math.round(displayH)

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (!detections.length || !video.videoWidth || !video.videoHeight) return

  // Capture dimensions (what we sent to the server)
  const captureH = Math.round(captureW * video.videoHeight / video.videoWidth)

  // Scale from capture space → display space
  const sx = displayW / captureW
  const sy = displayH / captureH

  detections.forEach(d => {
    const { top, right, bottom, left } = d.location
    const x = left * sx
    const y = top * sy
    const w = (right - left) * sx
    const h = (bottom - top) * sy

    const recognized = d.recognized
    const color = recognized ? '#22c55e' : '#ef4444'
    const bgColor = recognized ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.12)'

    // Face highlight box
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.strokeRect(x, y, w, h)
    ctx.fillStyle = bgColor
    ctx.fillRect(x, y, w, h)

    // Corner accents
    const cs = Math.min(w, h) * 0.18
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    // TL
    ctx.beginPath(); ctx.moveTo(x, y + cs); ctx.lineTo(x, y); ctx.lineTo(x + cs, y); ctx.stroke()
    // TR
    ctx.beginPath(); ctx.moveTo(x + w - cs, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cs); ctx.stroke()
    // BL
    ctx.beginPath(); ctx.moveTo(x, y + h - cs); ctx.lineTo(x, y + h); ctx.lineTo(x + cs, y + h); ctx.stroke()
    // BR
    ctx.beginPath(); ctx.moveTo(x + w - cs, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cs); ctx.stroke()

    // Label
    const label = d.label || 'Face'
    ctx.font = '600 13px "Segoe UI", system-ui'
    const textW = ctx.measureText(label).width
    const labelX = x
    const labelY = y > 32 ? y - 8 : y + h + 22
    ctx.fillStyle = color
    ctx.fillRect(labelX - 4, labelY - 17, textW + 12, 22)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, labelX + 2, labelY - 2)
  })
}

function clearCanvas(canvas) {
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatRate(v) { return (v == null) ? '—' : `${Number(v).toFixed(1)}%` }
function formatLastMarked(v) {
  if (!v) return 'Never'
  if (typeof v === 'string') return v
  return `${v.date} ${v.time}`
}
function formatDateTime(v) { return v ? v.replace('T', ' ') : '—' }
function extractError(err, fallback) { return err?.response?.data?.message || err?.message || fallback }
function localDate() {
  const n = new Date()
  return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, '0')}-${String(n.getDate()).padStart(2, '0')}`
}

export default App
