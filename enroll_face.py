import cv2
import os

def capture_faces(user_id, user_name):
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier('attendance/haarcascade_frontalface_default.xml')

    # Tạo thư mục dataset nếu chưa có
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    cam = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Không mở được camera!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f"dataset/{user_name}_{user_id}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Ảnh {count}/20", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Enroll Face', frame)
        if cv2.waitKey(1) == 27 or count >= 20:  # ESC hoặc đủ 20 ảnh
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Hoàn tất chụp ảnh cho:", user_name)
