from flask import Flask, render_template, request, redirect, url_for, flash
import os, cv2, datetime
import face_recognition
import sqlite3

app = Flask(__name__)
app.secret_key = "supersecret"
UPLOAD_FOLDER = 'static/photos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create DB
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            student_id TEXT UNIQUE,
            class_code TEXT,
            face_encoding BLOB
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            start_time TEXT,
            end_time TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            session_id INTEGER,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_all_encodings():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT student_id, name, face_encoding FROM students")
    data = c.fetchall()
    conn.close()

    known_encodings = []
    known_ids = []
    known_names = []
    for sid, name, enc in data:
        known_ids.append(sid)
        known_names.append(name)
        known_encodings.append(eval(enc))
    return known_encodings, known_ids, known_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        class_code = request.form['class_code']
        file = request.files['photo']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{student_id}.jpg")
        file.save(path)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            flash("Không tìm thấy khuôn mặt!")
            return redirect(url_for('register'))

        face_enc = encodings[0].tolist()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO students (name, student_id, class_code, face_encoding) VALUES (?, ?, ?, ?)",
                      (name, student_id, class_code, str(face_enc)))
            conn.commit()
            flash("Đăng ký thành công!")
        except sqlite3.IntegrityError:
            flash("MSSV đã tồn tại!")
        conn.close()
        return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/create-session', methods=['POST'])
def create_session():
    subject = request.form['subject']
    start = request.form['start_time']
    end = request.form['end_time']
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (subject, start_time, end_time) VALUES (?, ?, ?)", (subject, start, end))
    conn.commit()
    conn.close()
    flash("Tạo tiết học thành công!")
    return redirect(url_for('home'))

@app.route('/attendance/<int:session_id>', methods=['GET'])
def attendance(session_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = c.fetchone()
    conn.close()

    # Kiểm tra thời gian
    now = datetime.datetime.now()
    start_time = datetime.datetime.strptime(session[2], "%Y-%m-%dT%H:%M")
    end_time = datetime.datetime.strptime(session[3], "%Y-%m-%dT%H:%M")
    if not (start_time <= now <= end_time):
        return f"Chưa đến hoặc đã qua thời gian điểm danh cho môn {session[1]}!"

    # Mở webcam
    cap = cv2.VideoCapture(0)
    known_encodings, known_ids, known_names = get_all_encodings()
    marked = []

    while True:
        ret, frame = cap.read()
        rgb = frame[:, :, ::-1]
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for face_encoding, face_location in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                idx = matches.index(True)
                student_id = known_ids[idx]
                name = known_names[idx]

                if student_id not in marked:
                    marked.append(student_id)
                    conn = sqlite3.connect('database.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO attendance (student_id, session_id, timestamp) VALUES (?, ?, ?)",
                              (student_id, session_id, datetime.datetime.now().isoformat()))
                    conn.commit()
                    conn.close()

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} - {student_id}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Điểm danh kết thúc!"

if __name__ == '__main__':
    app.run(debug=True)
