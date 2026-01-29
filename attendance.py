import cv2
import numpy as np
import os
import time
from datetime import datetime

# --- Load model và cascade ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- Đọc file names.txt ---
names = {}
with open('names.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            id, name = parts
            names[int(id)] = name

# --- Khởi tạo camera ---
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

attendance_file = 'attendance_log.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', encoding='utf-8') as f:
        f.write('ID,Name,Time\n')

recognized = False
start_time = time.time()  # bắt đầu đếm thời gian
scan_duration = 3         # thời gian quét 3 giây

print(" Hệ thống đang quét khuôn mặt... Vui lòng giữ ổn định trước camera.")

while True:
    ret, img = cam.read()
    if not ret:
        print(" Không thể mở camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 80:
            name = names.get(id, "Unknown")
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(img, f"{name}", (x+5, y-5), font, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Nếu đã quét được khuôn mặt đủ 3 giây
            if time.time() - start_time >= scan_duration:
                with open(attendance_file, 'a', encoding='utf-8') as f:
                    f.write(f'{id},{name},{now}\n')

                print(f"✅ Chấm công thành công cho {name} lúc {now}")
                recognized = True
                break
        else:
            cv2.putText(img, "Khong nhan dien duoc, vui long giu mat gan camera", (x, y-10), font, 0.6, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Cham cong', img)

    # Khi đã nhận dạng xong thì hiển thị kết quả 1.5s rồi thoát
    if recognized:
        cv2.waitKey(1500)
        break

    # Thoát thủ công
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print(" Bạn đã thoát thủ công.")
        break

cam.release()
cv2.destroyAllWindows()
