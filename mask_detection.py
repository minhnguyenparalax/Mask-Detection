from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import csv
from datetime import datetime
from PIL import Image

# Khởi tạo các module detect mặt và facial landmark
face_detect = dlib.get_frontal_face_detector()
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Bắt đầu đọc từ camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Mở một file CSV để ghi thông tin
csv_file = open('output.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Mask Status', 'Image Path'])  # Ghi header

# Đường dẫn để lưu trữ hình ảnh
image_path = 'images/'

while True:
    # Đọc từ camera
    frame = vs.read()

    # Resize để tăng tốc độ xử lý
    frame = imutils.resize(frame, width=600)

    # Chuyển về ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect các khuôn mặt trong ảnh
    faces = face_detect(gray)

    # Biến để lưu trạng thái có đeo khẩu trang hay không
    mask_status = "Không"

    # Duyệt qua các khuôn mặt
    for rect in faces:
        # Nhận diện các điểm landmark
        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        # Capture vùng miệng
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        mouth = landmark[mStart:mEnd]

        # Lấy hình chữ nhật bao vùng miệng
        boundRect = cv2.boundingRect(mouth)
        cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 0, 255), 2)
        
        # Tính toán saturation trung bình
        hsv = cv2.cvtColor(frame[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]], cv2.COLOR_BGR2HSV)
        sum_saturation = np.sum(hsv[:, :, 1])
        area = boundRect[2] * boundRect[3]
        avg_saturation = sum_saturation / area

        # Kiểm tra và cập nhật trạng thái khẩu trang
        if avg_saturation > 100:
            mask_status = "Có"
        else:
            mask_status = "Không"
            # Vẽ thông báo lên màn hình
            cv2.putText(frame, "KHÔNG ĐEO KHAU TRANG!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Lưu lại ảnh của khuôn mặt không đeo khẩu trang
        if rect.top() < rect.bottom() and rect.left() < rect.right():
            timestamp_str = str(int(time.time()))
            image_filename = f"{timestamp_str}_no_mask.jpg"
            image_path_full = image_path + image_filename
            cv2.imwrite(image_path_full, frame[rect.top():rect.bottom(), rect.left():rect.right()])

            # Lấy timestamp hiện tại dưới dạng đối tượng datetime
            current_datetime = datetime.now()

            # Chuyển đổi định dạng và ghi thông tin vào file CSV
            formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([formatted_datetime, mask_status, image_path_full])

    # Hiển thị lên màn hình
    cv2.imshow("Camera", frame)

    # Bấm Esc để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    # Lưu ảnh mỗi 5 giây
    if int(time.time()) % 5 == 0:
    # Lưu ảnh dưới dạng định dạng PNG
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
        image_filename = f"{timestamp_str}_frame.png"
        image_path_full = image_path + image_filename
        cv2.imwrite(image_path_full, frame)


# Đóng file CSV
csv_file.close()

# Đóng cửa sổ hiển thị
cv2.destroyAllWindows()
vs.stop()