import sqlite3
import os
import cv2
import numpy as np
import uuid
import base64
import time
from io import BytesIO
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, Response, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import secrets

app = Flask(__name__)
# 🛑 SECURITY FIX: Random Secret Key
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# 🛑 SECURITY FIX: Session Cookie Protection
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# 🛑 SECURITY FIX: Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

INSTANCE_FOLDER = os.path.join(app.root_path, 'instance')
FACE_FOLDER = os.path.join(INSTANCE_FOLDER, 'student_faces')
LEAVE_DOC_FOLDER = os.path.join(INSTANCE_FOLDER, 'leave_documents') # NAYA: Leave documents ke liye
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(LEAVE_DOC_FOLDER, exist_ok=True) # NAYA

DB_NAME = "attendance.db"
CASCADE_PATH = os.path.join(app.root_path, 'static', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

ALL_SEMESTERS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
ALL_DEPARTMENTS = ["Computer Science", "English", "Urdu", "Physics", "Economics", "Mathematics", "Chemistry", "Biology", "Business Administration"]

def get_db():
    conn = sqlite3.connect(DB_NAME, timeout=20)
    return conn

def get_all_months():
    current_year = datetime.now().year
    months = []
    for year in [current_year - 1, current_year, current_year + 1]:
        for i in range(1, 13):
            months.append({
                "value": f"{year}-{str(i).zfill(2)}",
                "name": datetime(year, i, 1).strftime("%B")
            })
    months.reverse()
    return months

def format_month(month_str):
    try: return datetime.strptime(month_str, "%Y-%m").strftime("%B %Y")
    except: return month_str

def get_semester_session(semester):
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT start_date, end_date FROM semester_sessions WHERE semester=?", (semester,))
        row = c.fetchone()
        if row and row[0] and row[1]:
            return f"{row[0]} to {row[1]}"
    except Exception:
        pass
    finally:
        conn.close()
    return "Dates Not Set"

# ---------------- DATABASE ----------------
def init_db():
    conn = get_db()
    try:
        c = conn.cursor()
        # Users Table
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, roll_number TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL, department TEXT, semester TEXT, subject TEXT,
            password_hash TEXT NOT NULL, face_image TEXT, approval_status TEXT)""")
        
        # Attendance Table
        c.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, teacher_id INTEGER,
            department TEXT, semester TEXT, subject TEXT, session_name TEXT, date TEXT, time TEXT, status TEXT)""")
        
        # Class Sessions Table
        c.execute("""CREATE TABLE IF NOT EXISTS class_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, teacher_id INTEGER, department TEXT,
            semester TEXT, subject TEXT, session_name TEXT, date TEXT,
            UNIQUE(teacher_id, department, semester, subject, session_name, date))""")

        # Semester Sessions Table
        c.execute("""CREATE TABLE IF NOT EXISTS semester_sessions (
            semester TEXT PRIMARY KEY, start_date TEXT, end_date TEXT)""")
            
        # Leaves Table (Updated with document_path)
        c.execute("""CREATE TABLE IF NOT EXISTS leaves (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, department TEXT,
            semester TEXT, date TEXT, reason TEXT, document_path TEXT, status TEXT, applied_at TEXT)""")
            
        # TEACHER ASSIGNMENTS TABLE
        c.execute("""CREATE TABLE IF NOT EXISTS teacher_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, teacher_id INTEGER, department TEXT,
            semester TEXT, subject TEXT, session_name TEXT,
            UNIQUE(teacher_id, department, semester, subject, session_name))""")

        # Settings Table
        c.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('admin_code', '12345')")

        # Table for Dismissed Defaulters
        c.execute("""CREATE TABLE IF NOT EXISTS dismissed_defaulters (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, month TEXT)""")

        # Safe ALTER TABLE checks
        try:
            c.execute("ALTER TABLE semester_sessions ADD COLUMN start_date TEXT")
            c.execute("ALTER TABLE semester_sessions ADD COLUMN end_date TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            c.execute("ALTER TABLE users ADD COLUMN approval_status TEXT")
        except sqlite3.OperationalError:
            pass
            
        try:
            c.execute("ALTER TABLE users ADD COLUMN is_coordinator INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
        try:
            c.execute("ALTER TABLE leaves ADD COLUMN document_path TEXT")
        except sqlite3.OperationalError:
            pass
            
        for sem in ALL_SEMESTERS:
            c.execute("INSERT OR IGNORE INTO semester_sessions (semester, start_date, end_date) VALUES (?, NULL, NULL)", (sem,))
            
        try:
            c.execute("ALTER TABLE attendance ADD COLUMN session_name TEXT")
        except sqlite3.OperationalError:
            pass
            
        try:
            c.execute("ALTER TABLE class_sessions ADD COLUMN session_name TEXT")
        except sqlite3.OperationalError:
            pass
            
        try:
            c.execute("ALTER TABLE class_sessions ADD COLUMN section TEXT DEFAULT 'A'")
            c.execute("ALTER TABLE class_sessions ADD COLUMN lecture_number INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
                 
        conn.commit()
    finally:
        conn.close()

# ---------------- CONTEXT PROCESSOR ----------------
@app.context_processor
def inject_globals():
    data = {}
    if session.get('role') in ['hod', 'coordinator']:
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
            row = c.fetchone()
            if row:
                c.execute("""SELECT COUNT(*) FROM users 
                             WHERE role='teacher' AND department=? AND approval_status='pending'""",
                          (row[0],))
                data['pending_teacher_count'] = c.fetchone()[0]
            conn.close()
        except Exception:
            data['pending_teacher_count'] = 0
            
    if 'user_id' in session:
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT face_image FROM users WHERE id=?", (session['user_id'],))
            row = c.fetchone()
            if row and row[0]:
                data['current_user_image'] = row[0]
            conn.close()
        except Exception:
            pass
            
    return data

# ---------------- HELPERS ----------------
def add_title(ws, title, subtitle, ncols):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    tc = ws.cell(row=1, column=1, value=title)
    tc.font = Font(size=15, bold=True, color="1E3C72")
    tc.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    sc = ws.cell(row=2, column=1, value=subtitle)
    sc.font = Font(size=10, italic=True, color="64748B")
    sc.alignment = Alignment(horizontal="center")

def style_header(ws, headers, row=4):
    fill = PatternFill(start_color="1E3C72", end_color="1E3C72", fill_type="solid")
    font = Font(bold=True, color="FFFFFF", size=11)
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for i, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=i, value=h)
        c.fill = fill
        c.font = font
        c.border = border
        c.alignment = Alignment(horizontal="center", vertical="center")

def write_rows(ws, rows, start_row=5):
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for r_idx, row in enumerate(rows, start_row):
        for c_idx, val in enumerate(row, 1):
            c = ws.cell(row=r_idx, column=c_idx, value=val)
            c.border = border
            c.alignment = Alignment(horizontal="center", vertical="center")

def excel_response(wb, filename):
    out = BytesIO()
    wb.save(out)
    out.seek(0)
    return Response(out.read(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

def get_student_subjects(user_id, department, semester):
    conn = get_db()
    try:
        c = conn.cursor()
        subs = {}
        c.execute("""SELECT DISTINCT cs.subject, cs.teacher_id, u.name
                     FROM class_sessions cs LEFT JOIN users u ON cs.teacher_id = u.id
                     WHERE cs.department = ? AND cs.semester = ?""", (department, semester))
        for subj, tid, tname in c.fetchall():
            if subj:
                subs[subj] = {"subject": subj, "teacher_id": tid, "teacher_name": tname or "-"}
        c.execute("SELECT DISTINCT subject FROM attendance WHERE user_id = ? AND department = ? AND semester = ?", (user_id, department, semester))
        for (subj,) in c.fetchall():
            if subj and subj not in subs:
                subs[subj] = {"subject": subj, "teacher_id": None, "teacher_name": "-"}
        return list(subs.values())
    finally:
        conn.close()

def get_subject_stats(user_id, department, semester, subject, month=None):
    conn = get_db()
    try:
        c = conn.cursor()
        date_clause = " AND date LIKE ?" if month else ""
        
        p_total = [department, semester, subject]
        if month: p_total.append(f"{month}%")
        c.execute(f"SELECT COUNT(DISTINCT date) FROM class_sessions WHERE department=? AND semester=? AND subject=?{date_clause}", p_total)
        total = c.fetchone()[0]
        
        p_present = [user_id, subject]
        if month: p_present.append(f"{month}%")
        c.execute(f"SELECT COUNT(DISTINCT date) FROM attendance WHERE user_id=? AND subject=?{date_clause}", p_present)
        present = c.fetchone()[0]
        
        l_date_clause = " AND l.date LIKE ?" if month else ""
        p_leaves = [user_id, department, semester, subject]
        if month: p_leaves.append(f"{month}%")
        
        c.execute(f"""
            SELECT COUNT(DISTINCT l.date) FROM leaves l
            JOIN class_sessions cs ON l.date = cs.date AND cs.department = l.department AND cs.semester = l.semester
            WHERE l.student_id = ? AND l.status = 'Approved' 
            AND cs.department = ? AND cs.semester = ? AND cs.subject = ?{l_date_clause}
            AND l.date NOT IN (
                SELECT date FROM attendance WHERE user_id = ? AND subject = ?{date_clause}
            )
        """, p_leaves + [user_id, subject] + ([f"{month}%"] if month else []))
        leaves = c.fetchone()[0] or 0
        
        if total == 0 and present > 0: total = present + leaves
        absent = max(0, total - present - leaves)
        
        effective_total = total - leaves
        if effective_total > 0:
            pct = round((present / effective_total) * 100, 1)
        else:
            pct = 100.0 if present > 0 else 0.0
        
        if pct > 100.0: pct = 100.0
        
        return {"total": total, "present": present, "absent": absent, "leaves": leaves, "percentage": pct}
    finally:
        conn.close()

# ---------------- AUTH ----------------
@app.route("/")
def entry_portal():
    return render_template("entry.html")

@app.route("/login/<role>", methods=["GET", "POST"])
def login(role):
    if role not in ['student', 'teacher', 'hod']: 
        return redirect(url_for('entry_portal'))
        
    if request.method == "POST":
        if session.get('failed_attempts', 0) >= 5:
            last_attempt = session.get('last_attempt_time', 0)
            if time.time() - last_attempt < 60:
                flash("Too many failed attempts. Please wait 1 minute.", "danger")
                return render_template("login.html", role=role)

        roll = request.form.get("roll_number")
        pwd  = request.form.get("password")
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT id, name, password_hash, role, approval_status, is_coordinator FROM users WHERE roll_number = ?", (roll,))
            user = c.fetchone()
        finally:
            conn.close()
        
        if user and check_password_hash(user[2], pwd):
            actual_role = user[3]
            approval    = user[4]
            is_coord    = user[5]
            
            can_login = False
            if role == 'student' and actual_role == 'student':
                can_login = True
            elif role == 'teacher' and actual_role in ('teacher', 'hod'):
                can_login = True
            elif role == 'hod' and actual_role == 'hod':
                can_login = True
            
            if not can_login:
                flash(f"This account cannot login via {role} portal.", "danger")
                return render_template("login.html", role=role)
            
            if actual_role == 'teacher' and approval == 'pending':
                flash("Your registration request is pending. Please wait for your HOD to approve it.", "warning")
                return render_template("login.html", role=role)
                
            if approval == 'inactive':
                flash("Your account has been deactivated by the HOD. Please contact administration.", "danger")
                return render_template("login.html", role=role)
            
            session['failed_attempts'] = 0 
            session['user_id']     = user[0]
            session['user_name']   = user[1]
            
            # Coordinator Logic
            if actual_role == 'teacher' and is_coord == 1:
                session['role'] = 'coordinator'
            else:
                session['role'] = role
                
            session['actual_role'] = actual_role
            
            if session['role'] in ['teacher', 'coordinator']:  return redirect(url_for('teacher_portal'))
            if role == 'student':  return redirect(url_for('student_portal'))
            if role == 'hod':      return redirect(url_for('hod_portal'))
        
        session['failed_attempts'] = session.get('failed_attempts', 0) + 1
        session['last_attempt_time'] = time.time()
        flash("Invalid ID or Password!", "danger")
        
    return render_template("login.html", role=role)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('entry_portal'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        roll_number = request.form.get("roll_number")
        password = request.form.get("password")
        role = request.form.get("role")
        department = request.form.get("department")
        semester = request.form.get("semester")
        image_data = request.form.get("image_data")
        admin_code = request.form.get("admin_code", "").strip()

        conn_cfg = get_db()
        try:
            c_cfg = conn_cfg.cursor()
            c_cfg.execute("SELECT value FROM settings WHERE key=?", (f"admin_code_{department}",))
            dept_code_row = c_cfg.fetchone()
            if dept_code_row:
                SECRET_ADMIN_CODE = dept_code_row[0]
            else:
                c_cfg.execute("SELECT value FROM settings WHERE key='admin_code'")
                global_code_row = c_cfg.fetchone()
                SECRET_ADMIN_CODE = global_code_row[0] if global_code_row else "12345"
        finally:
            conn_cfg.close()

        if role == 'hod':
            if admin_code != SECRET_ADMIN_CODE:
                flash("Security Error: Invalid Admin Code for this department!", "danger")
                return redirect(url_for("register"))
            
            conn_check = get_db()
            try:
                c_check = conn_check.cursor()
                c_check.execute("SELECT COUNT(*) FROM users WHERE role='hod' AND department=?", (department,))
                if c_check.fetchone()[0] > 0:
                    flash(f"HOD already exists for {department} department!", "danger")
                    return redirect(url_for("register"))
            finally:
                conn_check.close()
        
        if role == 'student':
            if not semester or not department or not image_data:
                flash("Students must fill all fields and capture a photo!", "danger")
                return redirect(url_for("register"))
        elif role in ['teacher', 'hod']:
            semester = None 
            if not department:
                flash("Please select a Department!", "danger")
                return redirect(url_for("register"))
                
        filename, filepath = None, None
        if role == 'student' and image_data:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(FACE_FOLDER, filename)
            
        try:
            if role == 'student' and image_data:
                if len(image_data) > 5 * 1024 * 1024:
                    flash("Image is too large! Please try again with a normal photo.", "danger")
                    return redirect(url_for("register"))

                b64 = image_data.split(",")[1]
                img_bytes = base64.b64decode(b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    flash("Invalid image format! Please capture again.", "danger")
                    return redirect(url_for("register"))
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)

                attempts = [
                    (gray_eq,    1.10, 4, (25, 25), 1.0), 
                    (gray_eq,    1.10, 3, (20, 20), 1.0), 
                    (gray_eq,    1.05, 3, (15, 15), 1.0), 
                    (gray_clahe, 1.10, 3, (20, 20), 1.0), 
                    (gray_clahe, 1.05, 2, (15, 15), 1.0), 
                    (gray,       1.10, 3, (20, 20), 1.0), 
                    (gray,       1.05, 2, (15, 15), 1.0), 
                ]

                for upscale in [1.5, 2.0]:
                    up_eq    = cv2.resize(gray_eq,    None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
                    up_clahe = cv2.resize(gray_clahe, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
                    attempts.extend([
                        (up_eq,    1.10, 3, (int(30 * upscale), int(30 * upscale)), upscale), 
                        (up_clahe, 1.10, 2, (int(25 * upscale), int(25 * upscale)), upscale), 
                    ])

                faces = []
                for src_img, sf, mn, ms, up_r in attempts:
                    detected = face_cascade.detectMultiScale(src_img, scaleFactor=sf, minNeighbors=mn, minSize=ms)
                    if len(detected) > 0:
                        faces = [(int(x / up_r), int(y / up_r), int(w / up_r), int(h / up_r))
                                 for (x, y, w, h) in detected]
                        break

                if len(faces) == 0:
                    flash("No human face detected! Please ensure good lighting and come closer.", "danger")
                    return redirect(url_for("register"))

                if len(faces) > 1:
                    faces = list(faces)
                    faces.sort(key=lambda f: int(f[2]) * int(f[3]), reverse=True)
                    if int(faces[0][2]) * int(faces[0][3]) >= 2 * (int(faces[1][2]) * int(faces[1][3])):
                        faces = [faces[0]]
                    else:
                        flash("Multiple faces detected! Please capture alone.", "danger")
                        return redirect(url_for("register"))

                (x, y, w, h) = faces[0]
                face_roi = cv2.resize(gray_eq[y:y+h, x:x+w], (200, 200))

                conn_check = get_db()
                try:
                    c_check = conn_check.cursor()
                    c_check.execute("SELECT id, face_image, name, roll_number, semester FROM users WHERE role='student' AND face_image IS NOT NULL")
                    existing_users = c_check.fetchall()
                    
                    if existing_users:
                        temp_recognizer = cv2.face.LBPHFaceRecognizer_create()
                        temp_faces, temp_ids = [], []
                        for ex_id, ex_fname, ex_name, ex_roll, ex_sem in existing_users:
                            ex_path = os.path.join(FACE_FOLDER, ex_fname)
                            if os.path.exists(ex_path):
                                ex_img = cv2.imread(ex_path, cv2.IMREAD_GRAYSCALE)
                                if ex_img is not None:
                                    ex_img_eq = cv2.equalizeHist(ex_img)
                                    ex_faces = face_cascade.detectMultiScale(ex_img_eq, 1.1, 4, minSize=(20, 20))
                                    if len(ex_faces) == 0:
                                        ex_faces = face_cascade.detectMultiScale(ex_img_eq, 1.1, 3, minSize=(15, 15))
                                    if len(ex_faces) == 0:
                                        ex_faces = face_cascade.detectMultiScale(ex_img, 1.1, 3, minSize=(15, 15))
                                    if len(ex_faces) == 0:
                                        ex_clahe = clahe.apply(ex_img)
                                        ex_faces = face_cascade.detectMultiScale(ex_clahe, 1.1, 2, minSize=(15, 15))
                                    if len(ex_faces) > 0:
                                        (ex_x, ex_y, ex_w, ex_h) = ex_faces[0]
                                        temp_faces.append(cv2.resize(ex_img_eq[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w], (200, 200)))
                                        temp_ids.append(ex_id)
                                        
                        if temp_faces:
                            temp_recognizer.train(temp_faces, np.array(temp_ids))
                            label, confidence = temp_recognizer.predict(face_roi)
                            if confidence < 50: 
                                matched_user = next((u for u in existing_users if u[0] == label), None)
                                if matched_user:
                                    flash(f"ERROR: This face is already registered! (Roll: {matched_user[3]})", "danger")
                                    return redirect(url_for("register"))
                finally:
                    conn_check.close()
                    
                with open(filepath, "wb") as f: 
                    f.write(img_bytes)
                    
            approval = 'pending' if role == 'teacher' else 'approved'
            conn = get_db()
            try:
                c = conn.cursor()
                c.execute("""INSERT INTO users (name, roll_number, role, department, semester, password_hash, face_image, approval_status) 
                             VALUES (?,?,?,?,?,?,?,?)""", 
                          (name, roll_number, role, department, semester, generate_password_hash(password), filename, approval))
                conn.commit()
            finally:
                conn.close()
                
            if role == 'teacher':
                flash("Account submitted! Your HOD will review and approve it. You'll be able to login after approval.", "success")
            else:
                flash("Registration Successful! Please Login.", "success")
            return redirect(url_for("login", role=role))
            
        except sqlite3.IntegrityError:
            flash("This ID/Roll Number is already registered!", "warning")
            if role == 'student' and filename and os.path.exists(filepath): 
                os.remove(filepath)
            return redirect(url_for("register"))
            
    return render_template("register.html")

# ---------- PASSWORD CHANGE ----------
@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('entry_portal'))
    if request.method == "POST":
        current = request.form.get("current_password")
        new     = request.form.get("new_password")
        confirm = request.form.get("confirm_password")
        
        if new != confirm:
            flash("New passwords do not match!", "danger")
            return redirect(url_for('change_password'))
        if len(new) < 6:
            flash("Password must be at least 6 characters!", "warning")
            return redirect(url_for('change_password'))
        
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT password_hash FROM users WHERE id=?", (session['user_id'],))
            row = c.fetchone()
            if row and check_password_hash(row[0], current):
                c.execute("UPDATE users SET password_hash=? WHERE id=?",
                          (generate_password_hash(new), session['user_id']))
                conn.commit()
                flash("Password changed successfully!", "success")
            else:
                flash("Current password is incorrect!", "danger")
        finally:
            conn.close()
        return redirect(url_for('change_password'))
    return render_template("change_password.html")

# ---------- HOD & COORDINATOR ROUTES ----------
@app.route("/hod_teacher_requests")
def hod_teacher_requests():
    if session.get('role') not in ['hod', 'coordinator']:
        return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]
        c.execute("""SELECT id, name, roll_number FROM users 
                     WHERE role='teacher' AND department=? AND approval_status='pending'
                     ORDER BY name""", (dept,))
        pending = c.fetchall()
    finally:
        conn.close()
    return render_template("hod_teacher_requests.html", pending=pending, dept=dept)

@app.route("/approve_teacher/<int:teacher_id>", methods=["POST"])
def approve_teacher(teacher_id):
    if session.get('role') not in ['hod', 'coordinator']:
        return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        my_dept = c.fetchone()[0]
        c.execute("UPDATE users SET approval_status='approved' WHERE id=? AND role='teacher' AND department=?",
                  (teacher_id, my_dept))
        conn.commit()
        flash("Teacher approved! They can now login.", "success")
    finally:
        conn.close()
    return redirect(url_for('hod_teacher_requests'))

@app.route("/reject_teacher/<int:teacher_id>", methods=["POST"])
def reject_teacher(teacher_id):
    if session.get('role') not in ['hod', 'coordinator']:
        return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        my_dept = c.fetchone()[0]
        c.execute("DELETE FROM users WHERE id=? AND role='teacher' AND department=? AND approval_status='pending'",
                  (teacher_id, my_dept))
        conn.commit()
        flash("Teacher request rejected and removed.", "info")
    finally:
        conn.close()
    return redirect(url_for('hod_teacher_requests'))

@app.route("/make_coordinator/<int:teacher_id>", methods=["POST"])
def make_coordinator(teacher_id):
    if session.get('role') != 'hod': return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]

        c.execute("UPDATE users SET is_coordinator=0 WHERE department=?", (dept,))
        c.execute("UPDATE users SET is_coordinator=1 WHERE id=? AND role='teacher' AND department=?", (teacher_id, dept))
        conn.commit()
        flash("Teacher has been assigned as the new Department Coordinator!", "success")
    finally:
        conn.close()
    return redirect(url_for('hod_teachers'))

# ---------- HOD: SETTINGS (Secret Key) ----------
@app.route("/hod_settings", methods=["GET", "POST"])
def hod_settings():
    if session.get('role') != 'hod':
        return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        row = c.fetchone()
        if not row:
            return redirect(url_for('logout'))
        my_dept = row[0]
        
        setting_key = f"admin_code_{my_dept}"

        if request.method == "POST":
            current = request.form.get("current_code", "").strip()
            new     = request.form.get("new_code", "").strip()
            confirm = request.form.get("confirm_code", "").strip()
            
            if new != confirm:
                flash("New secret codes do not match!", "danger")
                return redirect(url_for('hod_settings'))
            if len(new) < 3:
                flash("Secret code must be at least 3 characters!", "warning")
                return redirect(url_for('hod_settings'))
            
            c.execute("SELECT value FROM settings WHERE key=?", (setting_key,))
            code_row = c.fetchone()
            
            if code_row:
                stored = code_row[0]
            else:
                c.execute("SELECT value FROM settings WHERE key='admin_code'")
                global_row = c.fetchone()
                stored = global_row[0] if global_row else "12345"
            
            if current != stored:
                flash("Current secret code is incorrect!", "danger")
            else:
                c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (setting_key, new))
                conn.commit()
                flash(f"Secret code for {my_dept} department updated successfully!", "success")
            return redirect(url_for('hod_settings'))
            
    finally:
        conn.close()
        
    return render_template("hod_settings.html")

# ---------------- STUDENT ROUTES ----------------
@app.route("/student_portal")
def student_portal():
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    
    month = request.args.get("month", "").strip()
    selected_subj = request.args.get("subject", "").strip()
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT name, roll_number, face_image, department, semester FROM users WHERE id=?", (session['user_id'],))
        info = c.fetchone()
        
        is_sem_active = False
        if info[4] != "Graduated":
            c.execute("SELECT start_date, end_date FROM semester_sessions WHERE semester=?", (info[4],))
            sess = c.fetchone()
            if sess and sess[0] and sess[1]:
                is_sem_active = True
                
    finally: conn.close()
    
    subjects = get_student_subjects(session['user_id'], info[3], info[4])
    subject_data = []
    
    for s in subjects:
        if selected_subj and s['subject'] != selected_subj:
            continue
        stats = get_subject_stats(session['user_id'], info[3], info[4], s['subject'], month or None)
        subject_data.append({**s, **stats})
        
    total_all = sum(s['total'] for s in subject_data)
    present_all = sum(s['present'] for s in subject_data)
    absent_all = sum(s['absent'] for s in subject_data)
    leaves_all = sum(s['leaves'] for s in subject_data)
    
    effective_total = total_all - leaves_all
    pct_all = round((present_all / effective_total) * 100, 1) if effective_total > 0 else 0
    if pct_all > 100: pct_all = 100.0
    
    return render_template("student_overview.html", student_info=info, 
                           total_all=total_all, present_all=present_all, 
                           absent_all=absent_all, leaves_all=leaves_all, pct_all=pct_all, 
                           all_months=get_all_months(), selected_month=month, 
                           all_subjects=subjects, selected_subj=selected_subj,
                           is_sem_active=is_sem_active)

@app.route("/student_subjects")
def student_subjects():
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT name, roll_number, face_image, department, semester FROM users WHERE id=?", (session['user_id'],))
        info = c.fetchone()
    finally: conn.close()
    
    subjects = get_student_subjects(session['user_id'], info[3], info[4])
    subject_data = []
    for s in subjects:
        stats = get_subject_stats(session['user_id'], info[3], info[4], s['subject'])
        subject_data.append({**s, **stats})
        
    return render_template("student_subjects.html", student_info=info, subject_data=subject_data)

@app.route("/student_leaves", methods=["GET", "POST"])
def student_leaves():
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department, semester FROM users WHERE id=?", (session['user_id'],))
        dept, sem = c.fetchone()
        
        if request.method == "POST":
            leave_date = request.form.get("leave_date")
            reason = request.form.get("reason")
            document = request.files.get("document") # NAYA
            
            if leave_date and reason:
                c.execute("SELECT id FROM leaves WHERE student_id=? AND date=? AND status != 'Rejected'", (session['user_id'], leave_date))
                if c.fetchone():
                   flash("You have already applied for leave on this date!", "danger")
                else:
                    doc_filename = None
                    # Document Upload Logic
                    if document and document.filename != '':
                        ext = os.path.splitext(document.filename)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.pdf']:
                            doc_filename = f"leave_{uuid.uuid4().hex}{ext}"
                            doc_path = os.path.join(LEAVE_DOC_FOLDER, doc_filename)
                            document.save(doc_path)
                        else:
                            flash("Invalid document format. Please upload JPG, PNG, or PDF.", "warning")
                            return redirect(url_for('student_leaves'))

                    c.execute("INSERT INTO leaves (student_id, department, semester, date, reason, document_path, status, applied_at) VALUES (?,?,?,?,?,?,?,?)", 
                              (session['user_id'], dept, sem, leave_date, reason, doc_filename, "Pending", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    flash("Leave application submitted successfully!", "success")
                return redirect(url_for('student_leaves'))
                
        c.execute("SELECT id, date, reason, applied_at, status, document_path FROM leaves WHERE student_id=? ORDER BY id DESC", (session['user_id'],))
        my_leaves = c.fetchall()
    finally: conn.close()
    return render_template("student_leaves.html", my_leaves=my_leaves)

@app.route("/delete_leave/<int:leave_id>", methods=["POST"])
def delete_leave(leave_id):
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT document_path FROM leaves WHERE id=? AND student_id=?", (leave_id, session['user_id']))
        row = c.fetchone()
        if row:
            # Document delete karein
            if row[0]:
                doc_path = os.path.join(LEAVE_DOC_FOLDER, row[0])
                if os.path.exists(doc_path): os.remove(doc_path)
            # Leave delete karein
            c.execute("DELETE FROM leaves WHERE id=? AND student_id=? AND status='Pending'", (leave_id, session['user_id']))
            conn.commit()
            flash("Leave application deleted.", "success")
    finally:
        conn.close()
    return redirect(url_for('student_leaves'))

@app.route("/leave_document/<filename>")
def leave_document(filename):
    if 'user_id' not in session: return "Unauthorized", 401
    if not filename or '/' in filename or '\\' in filename or '..' in filename:
        return "Forbidden", 403
    return send_from_directory(LEAVE_DOC_FOLDER, filename)

@app.route("/promote_semester")
def promote_semester():
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT semester FROM users WHERE id=?", (session['user_id'],))
        current_sem = c.fetchone()[0]

        if current_sem == "Graduated":
            flash("You have already Graduated.", "warning")
            return redirect(url_for('student_portal'))

        c.execute("SELECT start_date, end_date FROM semester_sessions WHERE semester=?", (current_sem,))
        row = c.fetchone()
        
        if row and row[0] and row[1]:
            flash(f"Action Denied! Your {current_sem} semester is currently active. You can only promote after the HOD deactivates this semester.", "danger")
            return redirect(url_for('student_portal'))

        sem_map = {"1st": "2nd", "2nd": "3rd", "3rd": "4th", "4th": "5th", "5th": "6th", "6th": "7th", "7th": "8th", "8th": "Graduated"}
        if current_sem in sem_map:
            next_sem = sem_map[current_sem]
            c.execute("UPDATE users SET semester=? WHERE id=?", (next_sem, session['user_id']))
            conn.commit()
            flash(f"Congratulations! You have been promoted to {next_sem} semester.", "success")
    finally: 
        conn.close()
    return redirect(url_for('student_portal'))

@app.route("/student_image/<filename>")
def student_image(filename):
    if 'user_id' not in session: 
        return "Unauthorized", 401
        
    role = session.get('role')
    user_id = session.get('user_id')

    if not filename or '/' in filename or '\\' in filename or '..' in filename:
        return "Forbidden", 403

    if role in ['teacher', 'hod', 'coordinator']:
        return send_from_directory(FACE_FOLDER, filename)
        
    if role == 'student':
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT face_image FROM users WHERE id=?", (user_id,))
            row = c.fetchone()
            if row and row[0] == filename:
                return send_from_directory(FACE_FOLDER, filename)
        finally:
            conn.close()
            
    return "Forbidden - You cannot access this image.", 403

@app.route("/update_face", methods=["GET", "POST"])
def update_face():
    if session.get('role') != 'student': return redirect(url_for('entry_portal'))
    
    if request.method == "POST":
        image_data = request.form.get("image_data")
        if not image_data:
            flash("No image captured!", "danger")
            return redirect(url_for('update_face'))
            
        try:
            b64 = image_data.split(",")[1]
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            
            faces = face_cascade.detectMultiScale(gray_eq, 1.1, 4, minSize=(25, 25))
            if len(faces) == 0:
                flash("No human face detected! Please try again in good lighting.", "danger")
                return redirect(url_for('update_face'))
                
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(FACE_FOLDER, filename)
            with open(filepath, "wb") as f: 
                f.write(img_bytes)
                
            conn = get_db()
            try:
                c = conn.cursor()
                c.execute("SELECT face_image FROM users WHERE id=?", (session['user_id'],))
                old_img = c.fetchone()
                if old_img and old_img[0]:
                    old_path = os.path.join(FACE_FOLDER, old_img[0])
                    if os.path.exists(old_path): os.remove(old_path)
                    
                c.execute("UPDATE users SET face_image=? WHERE id=?", (filename, session['user_id']))
                conn.commit()
            finally:
                conn.close()
                
            flash("Face data updated successfully!", "success")
            return redirect(url_for('student_portal'))
            
        except Exception as e:
            flash("An error occurred while processing the image.", "danger")
            return redirect(url_for('update_face'))

    return render_template("student_update_face.html")

# ---------------- TEACHER ROUTES ----------------
@app.route("/teacher_portal")
def teacher_portal():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        active_sessions = {}
        c.execute("SELECT semester, start_date, end_date FROM semester_sessions WHERE start_date IS NOT NULL AND end_date IS NOT NULL")
        for r in c.fetchall():
            active_sessions[r[0]] = f"{r[1]} to {r[2]}"
            
        c.execute("""
            SELECT ta.department, ta.semester, ta.subject, ta.session_name,
                   (SELECT COUNT(*) FROM users WHERE role='student' AND department=ta.department AND semester=ta.semester) as total_students,
                   (SELECT COUNT(DISTINCT date) FROM class_sessions WHERE department=ta.department AND semester=ta.semester AND subject=ta.subject AND session_name=ta.session_name) as lectures_taken,
                   (SELECT COUNT(*) FROM attendance WHERE department=ta.department AND semester=ta.semester AND subject=ta.subject AND session_name=ta.session_name) as total_presents
            FROM teacher_assignments ta
            WHERE ta.teacher_id = ?
            ORDER BY ta.department, ta.semester
        """, (session['user_id'],))
        
        raw_classes = c.fetchall()
        
        class_data = []
        total_lec_all = 0
        total_present_all = 0
        
        for dept, sem, subj, ta_session, stu_count, lec_count, present_count in raw_classes:
            if active_sessions.get(sem) == ta_session:
                total_lec_all += lec_count
                total_present_all += present_count
                
                total_possible = lec_count * stu_count
                pct = round((present_count / total_possible) * 100, 1) if total_possible > 0 else 0
                
                class_data.append({
                    "dept": dept, "sem": sem, "subj": subj,
                    "students": stu_count, "lectures": lec_count,
                    "present": present_count, "pct": pct
                })
            
    finally: conn.close()
    
    return render_template("teacher_overview.html", class_data=class_data, total_lec_all=total_lec_all, total_present_all=total_present_all)

@app.route("/teacher_class/<dept>/<sem>/<subj>")
def teacher_class_details(dept, sem, subj):
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        c.execute("SELECT id, name, roll_number, face_image FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (dept, sem))
        students = c.fetchall()
        
        student_details = []
        total_present = 0
        total_absent = 0
        total_leaves = 0
        
        for s_id, s_name, s_roll, s_img in students:
            stats = get_subject_stats(s_id, dept, sem, subj)
            total_present += stats['present']
            total_absent += stats['absent']
            total_leaves += stats['leaves']
            
            student_details.append({
                "id": s_id, "name": s_name, "roll": s_roll, "img": s_img,
                "total": stats['total'], "present": stats['present'], 
                "absent": stats['absent'], "leaves": stats['leaves'], "pct": stats['percentage']
            })
            
        c.execute("""SELECT date, session_name FROM class_sessions 
                     WHERE department=? AND semester=? AND subject=?
                     ORDER BY date DESC LIMIT 10""", (dept, sem, subj))
        sessions = c.fetchall()
        
        c.execute("SELECT COUNT(DISTINCT date) FROM class_sessions WHERE department=? AND semester=? AND subject=?", (dept, sem, subj))
        total_lectures = c.fetchone()[0]

    finally: conn.close()
    
    return render_template("teacher_class_details.html", 
                           dept=dept, sem=sem, subj=subj, 
                           students=student_details, sessions=sessions,
                           total_lectures=total_lectures,
                           total_present=total_present, total_absent=total_absent, total_leaves=total_leaves)

@app.route("/teacher_archived_reports")
def teacher_archived_reports():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        c.execute("""
            SELECT DISTINCT cs.department, cs.semester, cs.subject 
            FROM class_sessions cs
            JOIN semester_sessions ss ON cs.semester = ss.semester
            WHERE cs.teacher_id = ? AND (ss.start_date IS NULL OR ss.end_date IS NULL)
            ORDER BY cs.semester DESC, cs.subject
        """, (session['user_id'],))
        archived_classes = c.fetchall()
        
        class_details = []
        for dept, sem, subj in archived_classes:
            c.execute("SELECT COUNT(DISTINCT date) FROM class_sessions WHERE teacher_id=? AND department=? AND semester=? AND subject=?", (session['user_id'], dept, sem, subj))
            lec_count = c.fetchone()[0]
            
            c.execute("SELECT COUNT(*) FROM users WHERE role='student' AND department=? AND semester=?", (dept, sem))
            stu_count = c.fetchone()[0]
            
            class_details.append({
                "dept": dept, "sem": sem, "subj": subj, 
                "lectures": lec_count, "students": stu_count
            })
            
    finally: conn.close()
    
    return render_template("teacher_archived.html", class_details=class_details)

@app.route("/teacher_absentees")
def teacher_absentees():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    filter_date = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""SELECT DISTINCT department, semester, subject FROM class_sessions 
                     WHERE teacher_id=? AND date=?""", (session['user_id'], filter_date))
        classes = c.fetchall()
        
        absentees = []
        on_leaves = []
        
        for dept, sem, subj in classes:
            c.execute("SELECT id, name, roll_number FROM users WHERE role='student' AND department=? AND semester=?", (dept, sem))
            all_students = c.fetchall()
            
            c.execute("SELECT user_id FROM attendance WHERE date=? AND subject=? AND department=? AND semester=?", (filter_date, subj, dept, sem))
            present_ids = [r[0] for r in c.fetchall()]
            
            c.execute("SELECT student_id FROM leaves WHERE date=? AND department=? AND semester=? AND status='Approved'", (filter_date, dept, sem))
            leave_ids = [r[0] for r in c.fetchall()]
            
            for s_id, s_name, s_roll in all_students:
                if s_id in present_ids:
                    continue
                elif s_id in leave_ids:
                    on_leaves.append({"id": s_id, "dept": dept, "sem": sem, "subj": subj, "name": s_name, "roll": s_roll})
                else:
                    absentees.append({"id": s_id, "dept": dept, "sem": sem, "subj": subj, "name": s_name, "roll": s_roll})
                    
    finally: conn.close()
    
    return render_template("teacher_absentees.html", absentees=absentees, on_leaves=on_leaves, filter_date=filter_date)

@app.route("/mark_present_manual", methods=["POST"])
def mark_present_manual():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    s_id = request.form.get("student_id")
    dept = request.form.get("dept")
    sem = request.form.get("sem")
    subj = request.form.get("subj")
    date_str = request.form.get("date")
    
    active_session = get_semester_session(sem)
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT id FROM attendance WHERE user_id=? AND date=? AND subject=? AND department=? AND semester=? AND session_name=?", (s_id, date_str, subj, dept, sem, active_session))
        if not c.fetchone():
            time_now = datetime.now().strftime("%H:%M:%S")
            c.execute("INSERT INTO attendance (user_id, teacher_id, department, semester, subject, session_name, date, time, status) VALUES (?,?,?,?,?,?,?,?,?)", 
                      (s_id, session['user_id'], dept, sem, subj, active_session, date_str, time_now, "Present"))
            conn.commit()
            flash("Student manually marked as Present.", "success")
        else:
            flash("Student is already marked present for this class.", "info")
    finally:
        conn.close()
    return redirect(url_for('teacher_absentees', date=date_str))

@app.route("/teacher_attendance")
def teacher_attendance():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    pre_dept = request.args.get("dept", "")
    pre_sem = request.args.get("sem", "")
    pre_subj = request.args.get("subj", "")
    
    return render_template("teacher_attendance.html", 
                           all_semesters=ALL_SEMESTERS, 
                           all_departments=ALL_DEPARTMENTS,
                           pre_dept=pre_dept,
                           pre_sem=pre_sem,
                           pre_subj=pre_subj)

@app.route("/teacher_reports")
def teacher_reports():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))

    target_dept = request.args.get("department", "").strip()
    semester = request.args.get("semester", "").strip()
    subject = request.args.get("subject", "").strip()
    month = request.args.get("month", "").strip()

    class_stats = None
    error_msg = None

    if target_dept and semester and subject:
        conn = get_db()
        try:
            c = conn.cursor()
            active_session = get_semester_session(semester)
            
            if active_session == "Dates Not Set":
                error_msg = "This semester is currently inactive."
            else:
                c.execute("SELECT id FROM teacher_assignments WHERE teacher_id=? AND department=? AND semester=? AND LOWER(subject)=LOWER(?) AND session_name=?", 
                          (session['user_id'], target_dept, semester, subject, active_session))
                
                if not c.fetchone():
                    error_msg = f"Security Error: You are not assigned to teach {subject} in this class. You cannot view its analytics."
                else:
                    c.execute("SELECT id, name, roll_number FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (target_dept, semester))
                    students = c.fetchall()

                    date_clause = " AND date LIKE ?" if month else ""
                    p_total = [target_dept, semester, subject]
                    if month: p_total.append(f"{month}%")
                    c.execute(f"SELECT COUNT(DISTINCT date) FROM class_sessions WHERE department=? AND semester=? AND subject=?{date_clause}", p_total)
                    total_classes_held = c.fetchone()[0]

                    total_present = 0
                    total_absent = 0
                    total_leaves = 0
                    total_effective_classes = 0

                    for s_id, s_name, s_roll in students:
                        stats = get_subject_stats(s_id, target_dept, semester, subject, month or None)
                        total_present += stats['present']
                        total_absent += stats['absent']
                        total_leaves += stats['leaves']
                        total_effective_classes += (stats['total'] - stats['leaves'])

                    pct = round((total_present / total_effective_classes) * 100, 1) if total_effective_classes > 0 else 0
                    if pct > 100: pct = 100.0

                    class_stats = {
                        "dept": target_dept,
                        "sem": semester,
                        "subj": subject,
                        "month": month,
                        "students": len(students),
                        "total_classes": total_classes_held,
                        "present": total_present,
                        "absent": total_absent,
                        "leaves": total_leaves,
                        "pct": pct
                    }
        finally:
            conn.close()

    return render_template("teacher_reports.html", 
                           all_semesters=ALL_SEMESTERS, 
                           all_departments=ALL_DEPARTMENTS, 
                           all_months=get_all_months(),
                           class_stats=class_stats,
                           error_msg=error_msg,
                           filter_dept=target_dept,
                           filter_sem=semester,
                           filter_subj=subject,
                           filter_month=month)

@app.route("/teacher_leaves")
def teacher_leaves():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        teacher_dept = c.fetchone()[0]
        today = date.today().strftime("%Y-%m-%d")
        
        c.execute("""SELECT u.name, u.roll_number, l.semester, l.reason 
                     FROM leaves l JOIN users u ON l.student_id = u.id 
                     WHERE l.department = ? AND l.date = ? AND l.status != 'Rejected'
                     ORDER BY l.id DESC""", (teacher_dept, today))
        todays_leaves = c.fetchall()
        
        c.execute("""SELECT u.name, u.roll_number, l.semester, substr(l.date, 1, 7) as leave_month,
                            COUNT(l.id) as total_req,
                            SUM(CASE WHEN l.status = 'Approved' THEN 1 ELSE 0 END) as approved_count,
                            SUM(CASE WHEN l.status = 'Rejected' THEN 1 ELSE 0 END) as rejected_count
                     FROM leaves l JOIN users u ON l.student_id = u.id 
                     WHERE l.department = ? 
                     GROUP BY u.id, l.semester, leave_month
                     ORDER BY l.semester, leave_month DESC, u.name""", (teacher_dept,))
        raw_history = c.fetchall()
        leaves_history = [(h[0], h[1], h[2], format_month(h[3]), h[4], h[5], h[6]) for h in raw_history]
        
    finally: conn.close()
    return render_template("teacher_leaves.html", todays_leaves=todays_leaves, leaves_history=leaves_history)

@app.route("/teacher_students")
def teacher_students():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        active_sessions = {}
        c.execute("SELECT semester, start_date, end_date FROM semester_sessions WHERE start_date IS NOT NULL AND end_date IS NOT NULL")
        for r in c.fetchall():
            active_sessions[r[0]] = f"{r[1]} to {r[2]}"
            
        c.execute("SELECT department, semester, session_name FROM teacher_assignments WHERE teacher_id=?", (session['user_id'],))
        assigned_classes = c.fetchall()

        grouped_students = {}
        
        for dept, sem, ta_session in assigned_classes:
            if active_sessions.get(sem) == ta_session:
                c.execute("SELECT id, name, roll_number, face_image FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (dept, sem))
                students = c.fetchall()
                
                if students:
                    group_name = f"{dept} - {sem} Semester"
                    grouped_students[group_name] = students
                
    finally:
        conn.close()
        
    return render_template("teacher_students.html", grouped_students=grouped_students)

@app.route("/edit_student/<int:student_id>", methods=["GET", "POST"])
def edit_student(student_id):
    if session.get('role') not in ['teacher', 'hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        if request.method == "POST":
            new_name = request.form.get("name")
            new_roll = request.form.get("roll_number")
            new_password = request.form.get("password")
            
            if new_password:
                c.execute("UPDATE users SET name=?, roll_number=?, password_hash=? WHERE id=?", 
                          (new_name, new_roll, generate_password_hash(new_password), student_id))
            else:
                c.execute("UPDATE users SET name=?, roll_number=? WHERE id=?", (new_name, new_roll, student_id))
            conn.commit()
            flash("Student details updated successfully!", "success")
            return redirect(url_for('teacher_students'))
            
        c.execute("SELECT name, roll_number FROM users WHERE id=?", (student_id,))
        student = c.fetchone()
    finally:
        conn.close()
    return render_template("edit_student.html", student=student)

@app.route("/delete_student/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    if session.get('role') not in ['teacher', 'hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT face_image FROM users WHERE id=?", (student_id,))
        img_row = c.fetchone()
        if img_row and img_row[0]:
            img_path = os.path.join(FACE_FOLDER, img_row[0])
            if os.path.exists(img_path): os.remove(img_path)
        
        c.execute("DELETE FROM users WHERE id=? AND role='student'", (student_id,))
        conn.commit()
        flash("Student removed successfully.", "success")
    finally:
        conn.close()
    return redirect(url_for('teacher_students'))

@app.route("/reset_face/<int:student_id>", methods=["POST"])
def reset_face(student_id):
    if session.get('role') not in ['teacher', 'hod', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT face_image FROM users WHERE id=?", (student_id,))
        row = c.fetchone()
        
        if row and row[0]:
            img_path = os.path.join(FACE_FOLDER, row[0])
            if os.path.exists(img_path):
                os.remove(img_path)
                
        c.execute("UPDATE users SET face_image=NULL WHERE id=?", (student_id,))
        conn.commit()
        flash("Face data reset successfully. Student can now register a new face.", "success")
    finally:
        conn.close()
        
    return redirect(request.referrer or url_for('teacher_students'))

# ---------------- CAMERA LOGIC ----------------
@app.route("/start_attendance")
def start_attendance():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    target_dept = request.args.get("department")
    semester = request.args.get("semester")
    subject = request.args.get("subject", "").strip()
    
    if not target_dept or not semester or not subject:
        flash("Please select Department, Semester, and Subject.", "warning")
        return redirect(url_for('teacher_attendance'))
        
    conn = get_db()
    try:
        c = conn.cursor()
        
        active_session = get_semester_session(semester)
        if active_session == "Dates Not Set":
            flash(f"Error: {semester} Semester is currently inactive.", "danger")
            return redirect(url_for('teacher_attendance'))
            
        c.execute("""SELECT id FROM teacher_assignments 
                     WHERE teacher_id=? AND department=? AND semester=? AND LOWER(subject)=LOWER(?) AND session_name=?""", 
                  (session['user_id'], target_dept, semester, subject, active_session))
        if not c.fetchone():
            flash(f"Error: You are not assigned to teach {subject} in this class for the current session.", "danger")
            return redirect(url_for('teacher_attendance'))

        today = date.today().strftime("%Y-%m-%d")
        c.execute("""SELECT id FROM class_sessions 
                     WHERE department=? AND semester=? AND LOWER(subject)=LOWER(?) AND session_name=? AND date=?""", 
                  (target_dept, semester, subject, active_session, today))
        if c.fetchone():
            flash(f"Action Denied: Attendance for {subject} has already been taken today.", "warning")
            return redirect(url_for('teacher_attendance'))

        session['target_dept'], session['target_sem'], session['target_subj'], session['active_session'] = target_dept, semester, subject, active_session
        
        c.execute("INSERT OR IGNORE INTO class_sessions (teacher_id, department, semester, subject, session_name, date) VALUES (?,?,?,?,?,?)", (session['user_id'], target_dept, semester, subject, active_session, today))
        conn.commit()
        
        c.execute("SELECT name FROM users WHERE id=?", (session['user_id'],))
        t_name = c.fetchone()[0]
    finally: 
        conn.close()
        
    return render_template("live_attendance.html", teacher_name=t_name, target_dept=target_dept, target_sem=semester, target_subj=subject)

@app.route("/video_feed")
def video_feed():
    if session.get('role') not in ['teacher', 'coordinator']:
        return "Unauthorized", 401
    return Response(generate_frames(session.get('target_dept'), session.get('target_sem'), session.get('target_subj'), session.get('user_id'), session.get('active_session')), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/check_new_attendance")
def check_new_attendance():
    if session.get('role') not in ['teacher', 'coordinator']: return {"status": "error"}
    today = date.today().strftime("%Y-%m-%d")
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""SELECT a.id, u.name, u.roll_number, u.face_image, a.time FROM attendance a JOIN users u ON a.user_id = u.id 
                     WHERE a.teacher_id = ? AND a.date = ? AND a.subject = ? AND a.department = ? AND a.semester = ? AND a.session_name = ?
                     ORDER BY a.id DESC LIMIT 1""", (session.get('user_id'), today, session.get('target_subj'), session.get('target_dept'), session.get('target_sem'), session.get('active_session')))
        row = c.fetchone()
        if row: return {"status": "success", "id": row[0], "name": row[1], "roll": row[2], "image": row[3], "time": row[4]}
    finally: conn.close()
    return {"status": "none"}

def train_face_recognizer():
    try: recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError: return None
    faces, ids = [], []
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT id, face_image FROM users WHERE role='student' AND face_image IS NOT NULL")
        for uid, fname in c.fetchall():
            path = os.path.join(FACE_FOLDER, fname)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    detected = face_cascade.detectMultiScale(img, 1.3, 5)
                    if len(detected) == 0:
                        detected = face_cascade.detectMultiScale(img, 1.15, 4, minSize=(20, 20))
                    if len(detected) == 0:
                        detected = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(15, 15))
                    if len(detected) > 0:
                        (x, y, w, h) = detected[0]
                        faces.append(cv2.resize(img[y:y+h, x:x+w], (200, 200)))
                        ids.append(uid)
    finally: conn.close()
    if faces:
        recognizer.train(faces, np.array(ids))
        return recognizer
    return None

def generate_error_frame(text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (0, 0, 180)
    cv2.putText(img, "CAMERA ERROR", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, text, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', img)
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

def generate_frames(target_dept, target_sem, subject_name, teacher_id, active_session):
    if face_cascade.empty(): yield generate_error_frame("XML File Missing"); return
    recognizer = train_face_recognizer()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened(): yield generate_error_frame("Camera Locked"); return
    for _ in range(15): cap.read(); time.sleep(0.02)
    today, marked = date.today().strftime("%Y-%m-%d"), set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: yield generate_error_frame("Read Error"); break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            live_faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(live_faces) == 0:
                live_faces = face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(30, 30))
            if len(live_faces) == 0:
                live_faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(25, 25))
            for (x, y, w, h) in live_faces:
                color, label = (0, 0, 255), "Unknown"
                if recognizer:
                    id_, conf = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                    if conf < 65: 
                        conn = get_db()
                        try:
                            c = conn.cursor()
                            c.execute("SELECT name, department, semester FROM users WHERE id=?", (id_,))
                            u = c.fetchone()
                            if u and u[1] == target_dept and u[2] == target_sem:
                                label, color = f"{u[0]} (Verified)", (0, 255, 0)
                                if id_ not in marked:
                                    c.execute("SELECT id FROM attendance WHERE user_id=? AND date=? AND subject=? AND department=? AND semester=? AND session_name=?", (id_, today, subject_name, target_dept, target_sem, active_session))
                                    if not c.fetchone():
                                        c.execute("INSERT INTO attendance (user_id, teacher_id, department, semester, subject, session_name, date, time, status) VALUES (?,?,?,?,?,?,?,?,?)", (id_, teacher_id, target_dept, target_sem, subject_name, active_session, today, datetime.now().strftime("%H:%M:%S"), "Present"))
                                        conn.commit()
                                        marked.add(id_)
                            elif u: label, color = "WRONG CLASS!", (0, 0, 255)
                        finally: conn.close()
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()

# ---------------- EXCEL DOWNLOADS ----------------
@app.route("/download_teacher_report")
def download_teacher_report():
    if session.get('role') not in ['teacher', 'coordinator']: return redirect(url_for('entry_portal'))
    
    target_dept = request.args.get("department")
    semester = request.args.get("semester")
    subject = request.args.get("subject", "").strip()
    month = request.args.get("month", "").strip()
    
    active_session = get_semester_session(semester)
    if not target_dept or not semester or not subject: 
        flash("Select all fields.", "warning")
        return redirect(url_for('teacher_reports'))
        
    conn = get_db()
    try:
        c = conn.cursor()
        
        c.execute("SELECT id FROM teacher_assignments WHERE teacher_id=? AND department=? AND semester=? AND LOWER(subject)=LOWER(?) AND session_name=?", 
                  (session['user_id'], target_dept, semester, subject, active_session))
                  
        if not c.fetchone():
            flash("Security Error: You are not assigned to teach this subject. You cannot download its report.", "danger")
            return redirect(url_for('teacher_reports'))
            
        c.execute("SELECT name FROM users WHERE id=?", (session['user_id'],))
        t_name = c.fetchone()[0]
        c.execute("SELECT id, name, roll_number FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (target_dept, semester))
        students = c.fetchall()
    finally: 
        conn.close()
        
    wb = Workbook()
    ws = wb.active
    ws.title = "Class Report"
    headers = ["#", "Roll No", "Student Name", "Classes Held", "Present", "Absent", "Leaves", "Percentage"]
    add_title(ws, f"Class Report - {subject}", f"Teacher: {t_name} | Session: {active_session} | Month: {month or 'All Time'}", len(headers))
    style_header(ws, headers)
    
    rows = []
    for idx, (sid, sname, sroll) in enumerate(students, 1):
        stat = get_subject_stats(sid, target_dept, semester, subject, month or None)
        rows.append([idx, sroll, sname, stat['total'], stat['present'], stat['absent'], stat['leaves'], f"{stat['percentage']}%"])
        
    write_rows(ws, rows)
    for i, w in enumerate([6, 15, 25, 14, 10, 10, 10, 12], 1): 
        ws.column_dimensions[get_column_letter(i)].width = w
        
    return excel_response(wb, f"Class_{subject}_{semester}_{month or 'All'}.xlsx")

@app.route("/download_any_student_report/<int:student_id>")
def download_any_student_report(student_id):
    if session.get('role') not in ['teacher', 'hod', 'coordinator']: 
        flash("Unauthorized access! Only teachers and HODs can download reports.", "danger")
        return redirect(url_for('entry_portal'))
    
    month = request.args.get("month", "").strip()
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        if session.get('role') in ['teacher', 'coordinator']:
            c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
            my_dept = c.fetchone()[0]
            c.execute("SELECT department FROM users WHERE id=?", (student_id,))
            stu_dept = c.fetchone()[0]
            if my_dept != stu_dept:
                flash("Security Error: You cannot download reports for other departments.", "danger")
                return redirect(url_for('entry_portal'))

        c.execute("SELECT name, roll_number, department, semester FROM users WHERE id=?", (student_id,))
        st = c.fetchone()
        if not st:
            flash("Student not found.", "danger")
            return redirect(url_for('teacher_students'))
    finally: conn.close()
    
    subjects = get_student_subjects(student_id, st[2], st[3])
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    headers = ["Subject", "Teacher", "Classes Held", "Present", "Absent", "Leaves", "Percentage"]
    add_title(ws, "Attendance Report", f"Student: {st[0]} | Roll: {st[1]}", len(headers))
    style_header(ws, headers)
    rows = []
    for s in subjects:
        stat = get_subject_stats(student_id, st[2], st[3], s['subject'], month or None)
        rows.append([s['subject'], s['teacher_name'], stat['total'], stat['present'], stat['absent'], stat['leaves'], f"{stat['percentage']}%"])
    write_rows(ws, rows)
    for i, w in enumerate([25, 22, 14, 12, 12, 12, 12], 1): ws.column_dimensions[get_column_letter(i)].width = w
    safe_roll = st[1].replace(" ", "_")
    return excel_response(wb, f"Attendance_{safe_roll}_{month or 'All'}.xlsx")

# ---------------- HOD ROUTES ----------------
@app.route("/hod_portal")
def hod_portal():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE role='student' AND department=?", (dept,))
        total_students = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE role='teacher' AND department=?", (dept,))
        total_teachers = c.fetchone()[0]
        
        c.execute("SELECT id, name, roll_number, semester FROM users WHERE role='student' AND department=?", (dept,))
        all_students = c.fetchall()

        current_month = datetime.now().strftime("%Y-%m")
        
        c.execute("DELETE FROM dismissed_defaulters WHERE month != ?", (current_month,))
        conn.commit()
        
        c.execute("SELECT student_id FROM dismissed_defaulters WHERE month = ?", (current_month,))
        dismissed_ids = [row[0] for row in c.fetchall()]

        defaulters = []
        dept_total_classes, dept_total_present, dept_total_leaves = 0, 0, 0
        
        for sid, sname, sroll, ssem in all_students:
            subs = get_student_subjects(sid, dept, ssem)
            
            t_present = sum(get_subject_stats(sid, dept, ssem, s['subject'], current_month)['present'] for s in subs)
            t_total = sum(get_subject_stats(sid, dept, ssem, s['subject'], current_month)['total'] for s in subs)
            t_leaves = sum(get_subject_stats(sid, dept, ssem, s['subject'], current_month)['leaves'] for s in subs)
            
            dept_total_present += t_present
            dept_total_classes += t_total
            dept_total_leaves += t_leaves
            
            effective_total = t_total - t_leaves
            pct = round((t_present / effective_total) * 100, 1) if effective_total > 0 else 0
            if pct > 100: pct = 100.0
            
            if pct < 75 and effective_total > 0 and sid not in dismissed_ids: 
                defaulters.append({"id": sid, "name": sname, "roll": sroll, "sem": ssem, "pct": pct})
            
        defaulters_by_sem = {}
        for d in defaulters:
            if d['sem'] not in defaulters_by_sem:
                defaulters_by_sem[d['sem']] = []
            defaulters_by_sem[d['sem']].append(d)
            
        sorted_defaulters = {k: defaulters_by_sem[k] for k in sorted(defaulters_by_sem.keys(), key=lambda x: ALL_SEMESTERS.index(x) if x in ALL_SEMESTERS else 99)}

        effective_dept_total = dept_total_classes - dept_total_leaves
        dept_pct = round((dept_total_present / effective_dept_total) * 100, 1) if effective_dept_total > 0 else 0
        if dept_pct > 100: dept_pct = 100.0
        
        dept_absent = max(0, dept_total_classes - dept_total_present - dept_total_leaves)
    finally: 
        conn.close()
    
    return render_template("hod_overview.html", dept=dept, total_students=total_students, total_teachers=total_teachers, 
                           defaulters=sorted_defaulters, dept_present=dept_total_present, dept_absent=dept_absent, 
                           dept_pct=dept_pct, dept_leaves=dept_total_leaves, current_month=current_month)


@app.route("/dismiss_defaulter/<int:student_id>/<month>", methods=["POST"])
def dismiss_defaulter(student_id, month):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT id FROM dismissed_defaulters WHERE student_id=? AND month=?", (student_id, month))
        if not c.fetchone():
            c.execute("INSERT INTO dismissed_defaulters (student_id, month) VALUES (?, ?)", (student_id, month))
            conn.commit()
            flash("Student removed from the defaulter list for this month.", "success")
    finally:
        conn.close()
    return redirect(url_for('hod_portal'))


@app.route("/dismiss_semester_defaulters/<semester>/<month>", methods=["POST"])
def dismiss_semester_defaulters(semester, month):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]

        c.execute("SELECT id FROM users WHERE role='student' AND department=? AND semester=?", (dept, semester))
        students = c.fetchall()

        removed_count = 0
        for (sid,) in students:
            subs = get_student_subjects(sid, dept, semester)
            t_present = sum(get_subject_stats(sid, dept, semester, s['subject'], month)['present'] for s in subs)
            t_total = sum(get_subject_stats(sid, dept, semester, s['subject'], month)['total'] for s in subs)
            t_leaves = sum(get_subject_stats(sid, dept, semester, s['subject'], month)['leaves'] for s in subs)

            effective_total = t_total - t_leaves
            pct = round((t_present / effective_total) * 100, 1) if effective_total > 0 else 0

            if pct < 75 and effective_total > 0:
                c.execute("SELECT id FROM dismissed_defaulters WHERE student_id=? AND month=?", (sid, month))
                if not c.fetchone():
                    c.execute("INSERT INTO dismissed_defaulters (student_id, month) VALUES (?, ?)", (sid, month))
                    removed_count += 1
                    
        conn.commit()
        flash(f"Successfully removed {removed_count} defaulters from {semester} Semester.", "success")
    finally:
        conn.close()
    return redirect(url_for('hod_portal'))

@app.route("/hod_sessions")
def hod_sessions():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT semester, start_date, end_date 
            FROM semester_sessions 
            WHERE start_date IS NOT NULL 
            AND start_date != '' 
            AND end_date IS NOT NULL 
            AND end_date != ''
        """)
        raw_sessions = c.fetchall()
        sem_sessions = sorted(raw_sessions, key=lambda x: ALL_SEMESTERS.index(x[0]) if x[0] in ALL_SEMESTERS else 99)
    finally: conn.close()
    return render_template("hod_sessions.html", sem_sessions=sem_sessions, all_semesters=ALL_SEMESTERS)

@app.route("/hod_teachers")
def hod_teachers():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]
        
        c.execute("""SELECT id, name, roll_number, role, is_coordinator FROM users 
                     WHERE role IN ('teacher', 'hod') AND department=? 
                     AND (approval_status IS NULL OR approval_status != 'inactive')
                     ORDER BY role DESC, name""", (dept,))
        teachers = c.fetchall()
    finally: conn.close()
    return render_template("hod_teachers.html", teachers=teachers, dept=dept)

@app.route("/hod_assign_classes", methods=["GET", "POST"])
def hod_assign_classes():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        hod_dept = c.fetchone()[0]
        
        if request.method == "POST":
            teacher_id = request.form.get("teacher_id")
            semester = request.form.get("semester")
            subject = request.form.get("subject", "").strip()
            target_dept = hod_dept 
            
            if not teacher_id or not semester or not subject:
                flash("Please select a Teacher, Semester, and type a Subject Name!", "danger")
                return redirect(url_for('hod_assign_classes'))
            
            active_session = get_semester_session(semester)
            if active_session == "Dates Not Set":
                flash(f"Cannot assign class! {semester} Semester is currently inactive.", "danger")
            else:
                c.execute("""SELECT u.name FROM teacher_assignments ta
                             JOIN users u ON ta.teacher_id = u.id
                             WHERE ta.department = ? AND ta.semester = ? AND LOWER(ta.subject) = LOWER(?) AND ta.session_name = ?""", 
                          (target_dept, semester, subject, active_session))
                existing_assignment = c.fetchone()
                
                if existing_assignment:
                    flash(f"Error: '{subject}' is already assigned to {existing_assignment[0]} for this active session.", "danger")
                else:
                    try:
                        c.execute("""INSERT INTO teacher_assignments 
                                     (teacher_id, department, semester, subject, session_name) 
                                     VALUES (?,?,?,?,?)""", 
                                  (int(teacher_id), target_dept, semester, subject, active_session))
                        conn.commit()
                        flash("Class assigned successfully!", "success")
                    except sqlite3.IntegrityError:
                        flash("This class is already assigned to this teacher.", "warning")
            return redirect(url_for('hod_assign_classes'))
            
        c.execute("SELECT id, name, department, role FROM users WHERE role IN ('teacher','hod') AND (approval_status IS NULL OR approval_status != 'inactive') ORDER BY role DESC, name")
        all_teachers = c.fetchall()
        
        c.execute("SELECT semester, start_date, end_date FROM semester_sessions WHERE start_date IS NOT NULL AND end_date IS NOT NULL")
        active_sems = c.fetchall()
        active_semesters = sorted([r[0] for r in active_sems], key=lambda x: ALL_SEMESTERS.index(x) if x in ALL_SEMESTERS else 99)
        
        sem_counts = []
        for sem, sd, ed in active_sems:
            active_session = f"{sd} to {ed}"
            c.execute("SELECT COUNT(id) FROM teacher_assignments WHERE department = ? AND semester = ? AND session_name = ?", (hod_dept, sem, active_session))
            count = c.fetchone()[0]
            if count > 0:
                sem_counts.append((sem, count))
                
    finally: conn.close()
    
    return render_template("hod_assign_classes.html", 
                           all_teachers=all_teachers, 
                           active_semesters=active_semesters,
                           sem_counts=sem_counts,
                           hod_dept=hod_dept)

@app.route("/delete_teacher/<int:teacher_id>", methods=["POST"])
def delete_teacher(teacher_id):
    if session.get('role') != 'hod': 
        flash("Only HOD can delete teachers.", "danger")
        return redirect(url_for('entry_portal'))
    if teacher_id == session['user_id']:
        flash("You cannot delete yourself!", "danger")
        return redirect(url_for('hod_teachers'))
        
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE id=?", (teacher_id,))
        row = c.fetchone()
        
        if row and row[0] == 'hod':
            flash("HOD account cannot be deleted from here.", "danger")
            return redirect(url_for('hod_teachers'))
            
        c.execute("SELECT COUNT(*) FROM attendance WHERE teacher_id=?", (teacher_id,))
        has_attendance = c.fetchone()[0] > 0
        
        if has_attendance:
            c.execute("UPDATE users SET approval_status='inactive' WHERE id=?", (teacher_id,))
            c.execute("DELETE FROM teacher_assignments WHERE teacher_id=?", (teacher_id,))
            flash("Teacher has been deactivated. Their historical data is safe, and they can no longer login.", "success")
        else:
            c.execute("DELETE FROM teacher_assignments WHERE teacher_id=?", (teacher_id,))
            c.execute("DELETE FROM users WHERE id=? AND role='teacher'", (teacher_id,))
            flash("Teacher removed completely (No attendance history found).", "success")
            
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for('hod_teachers'))

@app.route("/hod_semester_details/<semester>")
def hod_semester_details(semester):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        hod_dept = c.fetchone()[0]
        
        active_session = get_semester_session(semester)
        if active_session == "Dates Not Set":
            assignments = []
            flash(f"{semester} semester is currently inactive. Please activate it to see assignments.", "warning")
        else:
            c.execute("""SELECT ta.semester, ta.subject, u.name, u.department, ta.id 
                         FROM teacher_assignments ta
                         JOIN users u ON ta.teacher_id = u.id
                         WHERE ta.department = ? AND ta.semester = ? AND ta.session_name = ?
                         ORDER BY ta.subject""", (hod_dept, semester, active_session))
            assignments = c.fetchall()
    finally: conn.close()
    
    return render_template("hod_semester_details.html", assignments=assignments, semester=semester, hod_dept=hod_dept)

@app.route("/delete_assignment/<int:assignment_id>", methods=["POST"])
def delete_assignment(assignment_id):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM teacher_assignments WHERE id=?", (assignment_id,))
        conn.commit()
        flash("Class assignment deleted successfully.", "success")
    finally:
        conn.close()
    return redirect(request.referrer or url_for('hod_assign_classes'))

@app.route("/edit_assignment/<int:assignment_id>", methods=["POST"])
def edit_assignment(assignment_id):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    new_subject = request.form.get("subject", "").strip()
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department, semester, session_name FROM teacher_assignments WHERE id=?", (assignment_id,))
        ta = c.fetchone()
        
        if ta:
            dept, sem, session_name = ta[0], ta[1], ta[2]
            c.execute("""SELECT u.name FROM teacher_assignments ta
                         JOIN users u ON ta.teacher_id = u.id
                         WHERE ta.department = ? AND ta.semester = ? AND LOWER(ta.subject) = LOWER(?) AND ta.id != ? AND ta.session_name = ?""", 
                      (dept, sem, new_subject, assignment_id, session_name))
            existing = c.fetchone()
            
            if existing:
                flash(f"Error: '{new_subject}' is already assigned to {existing[0]}.", "danger")
            else:
                c.execute("UPDATE teacher_assignments SET subject=? WHERE id=?", (new_subject, assignment_id))
                conn.commit()
                flash("Class assignment updated successfully!", "success")
        else:
            flash("Assignment not found.", "danger")
    finally:
        conn.close()
    return redirect(request.referrer or url_for('hod_assign_classes'))

@app.route("/hod_leaves")
def hod_leaves():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]
        
        c.execute("""SELECT l.id, u.name, u.roll_number, l.semester, l.date, l.reason, l.applied_at, l.document_path 
                     FROM leaves l JOIN users u ON l.student_id = u.id 
                     WHERE l.department = ? AND l.status = 'Pending' 
                     ORDER BY l.id DESC""", (dept,))
        pending_leaves = c.fetchall()
        
        c.execute("""SELECT u.name, u.roll_number, l.semester, substr(l.date, 1, 7) as leave_month,
                            COUNT(l.id) as total_req,
                            SUM(CASE WHEN l.status = 'Approved' THEN 1 ELSE 0 END) as approved_count,
                            SUM(CASE WHEN l.status = 'Rejected' THEN 1 ELSE 0 END) as rejected_count
                     FROM leaves l JOIN users u ON l.student_id = u.id 
                     WHERE l.department = ? 
                     GROUP BY u.id, l.semester, leave_month
                     ORDER BY l.semester, leave_month DESC, u.name""", (dept,))
        raw_summary = c.fetchall()
        
        leave_summary = [(s[0], s[1], s[2], format_month(s[3]), s[4], s[5], s[6]) for s in raw_summary]
        
    finally: conn.close()
    return render_template("hod_leaves.html", pending_leaves=pending_leaves, leave_summary=leave_summary, all_semesters=ALL_SEMESTERS)

@app.route("/update_leave_status/<int:leave_id>/<status>", methods=["POST"])
def update_leave_status(leave_id, status):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    if status in ['Approved', 'Rejected']:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("UPDATE leaves SET status=? WHERE id=?", (status, leave_id))
            conn.commit()
            flash(f"Leave request has been {status} successfully.", "success")
        finally: conn.close()
    return redirect(url_for('hod_leaves'))

@app.route("/update_semester_session", methods=["POST"])
def update_semester_session():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    sem, start_date, end_date = request.form.get("semester"), request.form.get("start_date", "").strip(), request.form.get("end_date", "").strip()
    if sem and start_date and end_date:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("UPDATE semester_sessions SET start_date=?, end_date=? WHERE semester=?", (start_date, end_date, sem))
            conn.commit()
            flash(f"{sem} Semester dates updated successfully!", "success")
        finally: conn.close()
    return redirect(url_for('hod_sessions'))

@app.route("/clear_semester_session/<semester>", methods=["POST"])
def clear_semester_session(semester):
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("UPDATE semester_sessions SET start_date=NULL, end_date=NULL WHERE semester=?", (semester,))
        conn.commit()
        flash(f"{semester} Semester deactivated.", "info")
    finally: conn.close()
    return redirect(url_for('hod_sessions'))

@app.route("/hod_report")
def hod_report():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]
        filter_sem, filter_month = request.args.get("semester", ""), request.args.get("month", "")
        report_data, subjects_list = [], []
        
        if filter_sem and filter_month:
            c.execute("SELECT DISTINCT subject FROM class_sessions WHERE department=? AND semester=?", (dept, filter_sem))
            subjects_list = [r[0] for r in c.fetchall() if r[0]]
            
            c.execute("""SELECT DISTINCT a.subject FROM attendance a JOIN users u ON a.user_id=u.id
                         WHERE u.department=? AND u.semester=?""", (dept, filter_sem))
            for (s,) in c.fetchall():
                if s and s not in subjects_list:
                    subjects_list.append(s)

            c.execute("SELECT id, name, roll_number FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (dept, filter_sem))
            students = c.fetchall()
            for sid, sname, sroll in students:
                row = {"id": sid, "name": sname, "roll": sroll, "subjects": {}}
                for subj in subjects_list: 
                    row["subjects"][subj] = get_subject_stats(sid, dept, filter_sem, subj, filter_month)
                report_data.append(row)
    finally: conn.close()
    
    return render_template("hod_report.html", hod_dept=dept, filter_sem=filter_sem, filter_month=filter_month, 
                           report_data=report_data, subjects_list=subjects_list, all_semesters=ALL_SEMESTERS, 
                           all_months=get_all_months())

@app.route("/student_history/<int:student_id>")
def student_history(student_id):
    if session.get('role') not in ['hod', 'teacher', 'coordinator']: 
        return redirect(url_for('entry_portal'))
        
    if session.get('role') in ['teacher', 'hod', 'coordinator']:
        conn_chk = get_db()
        try:
            c_chk = conn_chk.cursor()
            c_chk.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
            my_dept = c_chk.fetchone()[0]
            
            c_chk.execute("SELECT department FROM users WHERE id=?", (student_id,))
            stu_dept = c_chk.fetchone()[0]
            
            if my_dept != stu_dept:
                flash("Security Error: You cannot view students from other departments.", "danger")
                return redirect(url_for('entry_portal'))
        finally:
            conn_chk.close()
            
    month = request.args.get("month", "").strip()
    subject = request.args.get("subject", "").strip()
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT name, roll_number, department, semester, face_image FROM users WHERE id=?", (student_id,))
        student_info = c.fetchone()
        
        if not student_info:
            flash("Student not found.", "danger")
            return redirect(url_for('hod_report'))
            
        query = "SELECT subject, date, time, status FROM attendance WHERE user_id=?"
        params = [student_id]
        
        if month:
            query += " AND date LIKE ?"
            params.append(f"{month}%")
        if subject:
            query += " AND subject = ?"
            params.append(subject)
            
        query += " ORDER BY date DESC, time DESC"
        c.execute(query, params)
        history_data = c.fetchall()
    finally:
        conn.close()
        
    return render_template("student_history.html", 
                           student_info=student_info, 
                           student_id=student_id,
                           history_data=history_data, 
                           all_months=get_all_months(), 
                           filter_month=month, 
                           filter_subject=subject)

@app.route("/download_hod_report")
def download_hod_report():
    if session.get('role') not in ['hod', 'coordinator']: return redirect(url_for('entry_portal'))
    semester = request.args.get("semester", "").strip()
    month = request.args.get("month", "").strip()
    
    if not semester:
        flash("Please select a semester.", "warning")
        return redirect(url_for('hod_report'))

    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT department FROM users WHERE id=?", (session['user_id'],))
        dept = c.fetchone()[0]

        c.execute("SELECT DISTINCT subject FROM class_sessions WHERE department=? AND semester=?", (dept, semester))
        subjects = [r[0] for r in c.fetchall() if r[0]]
        c.execute("""SELECT DISTINCT a.subject FROM attendance a JOIN users u ON a.user_id=u.id
                     WHERE u.department=? AND u.semester=?""", (dept, semester))
        for (s,) in c.fetchall():
            if s and s not in subjects:
                subjects.append(s)

        c.execute("SELECT id, name, roll_number FROM users WHERE role='student' AND department=? AND semester=? ORDER BY roll_number", (dept, semester))
        students = c.fetchall()
    finally:
        conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    
    title = f"HOD Report - {dept} - {semester} Semester"
    if month:
        try: title += f" - {datetime.strptime(month, '%Y-%m').strftime('%B %Y')}"
        except: pass
    subtitle = f"Total Students: {len(students)} | Subjects: {len(subjects)}"

    headers = ["#", "Roll No", "Name"] + subjects + ["Overall %"]
    add_title(ws, title, subtitle, len(headers))
    style_header(ws, headers)

    rows = []
    for idx, (sid, sname, sroll) in enumerate(students, 1):
        row = [idx, sroll, sname]
        tp = tt = 0
        for subj in subjects:
            stat = get_subject_stats(sid, dept, semester, subj, month or None)
            row.append(f"{stat.get('present', 0)}/{stat.get('total', 0)}")
            tp += stat.get('present', 0)
            tt += stat.get('total', 0)
            
        pct = round((tp / tt) * 100, 1) if tt > 0 else 0
        if pct > 100: pct = 100.0
        row.append(f"{pct}%")
        rows.append(row)
        
    write_rows(ws, rows)
    for i, w in enumerate([6, 15, 22] + [12] * len(subjects) + [12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    for subj in subjects:
        safe_name = (subj[:28] + "..." if len(subj) > 31 else subj) or "Subject"
        try:
            ws_sub = wb.create_sheet(title=safe_name)
        except Exception:
            ws_sub = wb.create_sheet(title=safe_name + "_2")
            
        h2 = ["#", "Roll No", "Name", "Classes Held", "Present", "Absent", "Leaves", "Percentage"]
        add_title(ws_sub, f"{subj} - {semester}", f"{dept}" + (f" | {month}" if month else ""), len(h2))
        style_header(ws_sub, h2)
        
        srows = []
        for idx, (sid, sname, sroll) in enumerate(students, 1):
            stat = get_subject_stats(sid, dept, semester, subj, month or None)
            srows.append([idx, sroll, sname, stat.get('total', 0), stat.get('present', 0), stat.get('absent', 0), stat.get('leaves', 0), f"{stat.get('percentage', 0)}%"])
            
        write_rows(ws_sub, srows)
        for i, w in enumerate([6, 15, 25, 14, 10, 10, 10, 12], 1):
            ws_sub.column_dimensions[get_column_letter(i)].width = w

    safe_dept = dept.replace(" ", "_")
    return excel_response(wb, f"HOD_{safe_dept}_{semester}_{month or 'All'}.xlsx")

@app.route("/my_profile", methods=["GET", "POST"])
def my_profile():
    if session.get('role') not in ['teacher', 'hod', 'coordinator']: return redirect(url_for('entry_portal'))
    
    conn = get_db()
    try:
        c = conn.cursor()
        if request.method == "POST":
            new_name = request.form.get("name")
            image_file = request.files.get("profile_pic")
            
            c.execute("UPDATE users SET name=? WHERE id=?", (new_name, session['user_id']))
            session['user_name'] = new_name
            
            if image_file and image_file.filename != '':
                c.execute("SELECT face_image FROM users WHERE id=?", (session['user_id'],))
                old_img = c.fetchone()
                if old_img and old_img[0]:
                    old_path = os.path.join(FACE_FOLDER, old_img[0])
                    if os.path.exists(old_path): os.remove(old_path)
                
                filename = f"user_{uuid.uuid4().hex}.jpg"
                filepath = os.path.join(FACE_FOLDER, filename)
                image_file.save(filepath)
                c.execute("UPDATE users SET face_image=? WHERE id=?", (filename, session['user_id']))
                
            conn.commit()
            flash("Profile updated successfully!", "success")
            return redirect(url_for('my_profile'))
            
        c.execute("SELECT name, roll_number, department, face_image, role FROM users WHERE id=?", (session['user_id'],))
        user_info = c.fetchone()
    finally:
        conn.close()
        
    return render_template("my_profile.html", user_info=user_info)

if __name__ == "__main__":
    init_db()
    # 🛑 SECURITY FIX: Debug mode must be False in production/public URLs
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)