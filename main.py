import face_recognition
import cv2
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Tạo thư mục nếu chưa có
os.makedirs(DATASET_DIR, exist_ok=True)

def add_student():
    name = input("Nhập tên sinh viên: ").strip()
    cam = cv2.VideoCapture(0)
    print("Đang chụp khuôn mặt... Nhấn 's' để lưu, 'q' để thoát.")
    while True:
        ret, frame = cam.read()
        cv2.imshow("Thêm sinh viên", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
            cv2.imwrite(filepath, frame)
            print("Đã lưu ảnh.")
            break
        elif key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def load_known_faces():
    known_faces, known_names = [], []
    for filename in os.listdir(DATASET_DIR):
        img_path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names

def start_attendance():
    known_faces, known_names = load_known_faces()
    if not known_faces:
        print("Chưa có sinh viên nào được thêm.")
        return

    cam = cv2.VideoCapture(0)
    print("Bắt đầu điểm danh... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cam.read()
        rgb = frame[:, :, ::-1]
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for enc, loc in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_faces, enc)
            name = "Unknown"
            if True in matches:
                index = matches.index(True)
                name = known_names[index]
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now])
                print(f"Điểm danh: {name} lúc {now}")

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        cv2.imshow("Điểm danh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def view_statistics():
    if not os.path.exists(ATTENDANCE_FILE):
        print("Chưa có dữ liệu điểm danh.")
        return

    data = {}
    with open(ATTENDANCE_FILE, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            name = row[0]
            data[name] = data.get(name, 0) + 1

    print("=== Thống kê điểm danh ===")
    for name, count in data.items():
        print(f"{name}: {count} lần")

def plot_statistics():
    if not os.path.exists(ATTENDANCE_FILE):
        print("Chưa có dữ liệu điểm danh.")
        return

    data = {}
    with open(ATTENDANCE_FILE, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            name = row[0]
            data[name] = data.get(name, 0) + 1

    if not data:
        print("Không có dữ liệu để vẽ.")
        return

    names = list(data.keys())
    counts = list(data.values())
    plt.figure(figsize=(10,5))
    plt.bar(names, counts)
    plt.title("Số lần điểm danh của sinh viên")
    plt.xlabel("Tên sinh viên")
    plt.ylabel("Số lần có mặt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n=== Attendance System Menu ===")
        print("1. Add new student (Capture from Webcam)")
        print("2. Start attendance session")
        print("3. View attendance statistics")
        print("4. Plot attendance statistics")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            add_student()
        elif choice == '2':
            start_attendance()
        elif choice == '3':
            view_statistics()
        elif choice == '4':
            plot_statistics()
        elif choice == '5':
            break
        else:
            print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
