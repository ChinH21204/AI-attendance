from flask import Flask, render_template, request
import cv2, os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll_page')
def enroll_page():
    return render_template('enrolls.html')

@app.route('/enroll_face', methods=['POST'])
def enroll_face():
    emp_id = request.form['emp_id']
    emp_name = request.form['emp_name']

    # Tạo thư mục dataset nếu chưa có
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Mở camera
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            # Lưu ảnh khuôn mặt
            cv2.imwrite(f"{dataset_dir}/{emp_name}_{emp_id}_{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Đang chụp khuôn mặt...', img)
        # Nhấn ESC hoặc chụp 10 ảnh thì dừng
        if cv2.waitKey(100) & 0xFF == 27:
            break
        elif count >= 10:
            break

    cam.release()
    cv2.destroyAllWindows()

    return "✅ Đăng ký khuôn mặt thành công!"

if __name__ == "__main__":
    print("✅ Flask server chạy OK! Mở: http://127.0.0.1:5000/")
    app.run(debug=True)
