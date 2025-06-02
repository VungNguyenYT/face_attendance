# Cài đặt thư viện cần thiết
#pip install face_recognition
#pip install opencv-python

import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
from google.colab import files

# 1. Upload ảnh sinh viên cần nhận diện
uploaded = files.upload()

# 2. Đọc và encode ảnh sinh viên
known_faces = []
known_names = []

for filename in uploaded.keys():
    image = face_recognition.load_image_file(filename)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(filename.split('.')[0])

# 3. Upload ảnh lớp học (ảnh chứa nhiều sinh viên)
print("Upload ảnh cần điểm danh...")
uploaded_class = files.upload()
test_img_path = list(uploaded_class.keys())[0]
test_image = face_recognition.load_image_file(test_img_path)
test_face_locations = face_recognition.face_locations(test_image)
test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)

# 4. So sánh và ghi điểm danh
attendance_list = []

for encoding, location in zip(test_face_encodings, test_face_locations):
    matches = face_recognition.compare_faces(known_faces, encoding)
    name = "Unknown"
    if True in matches:
        index = matches.index(True)
        name = known_names[index]
        attendance_list.append((name, str(datetime.now())))

# 5. In kết quả
print("Kết quả điểm danh:")
for name, time in attendance_list:
    print(f"{name} - {time}")
