from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import csv
from datetime import datetime

# Khoi tao cac module detect mat va facial landmark
face_detect = dlib.get_frontal_face_detector()
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Doc tu camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Mở một file CSV để ghi thông tin
csv_file = open('output.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Mask Status'])  # Ghi header

while True:
    # Doc tu camera
    frame = vs.read()

    # Resize de tang toc do xu ly
    frame = imutils.resize(frame, width=600)

    # Chuyen ve gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cac mat trong anh
    faces = face_detect(gray)

    # Biến để lưu trạng thái có đeo khẩu trang hay không
    mask_status = "Không"

    # Duyet qua cac mat
    for rect in faces:
        # Nhan dien cac diem landmark
        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        # Capture vung mieng
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        mouth = landmark[mStart:mEnd]

        # Lay hinh chu nhat bao vung mieng
        boundRect = cv2.boundingRect(mouth)
        cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 0, 255), 2)

        # Tinh toan saturation trung binh
        hsv = cv2.cvtColor(frame[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]], cv2.COLOR_BGR2HSV)
        sum_saturation = np.sum(hsv[:, :, 1])
        area = boundRect[2] * boundRect[3]
        avg_saturation = sum_saturation / area

        # Kiểm tra và cập nhật trạng thái khẩu trang
        if avg_saturation > 100:
            mask_status = "Có"
            cv2.putText(frame, "DEO KHAU TRANG VAO! TOANG BAY GIO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
     # Lấy timestamp hiện tại dưới dạng đối tượng datetime
    current_datetime = datetime.now()
    
     # Chuyển đổi định dạng và ghi thông tin vào file CSV
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    csv_writer.writerow([formatted_datetime, mask_status])
    # Hien thi len man hinh
    cv2.imshow("Camera", frame)

    # Bam Esc de thoat
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Đóng file CSV
csv_file.close()

cv2.destroyAllWindows()
vs.stop()
