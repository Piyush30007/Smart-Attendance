import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    print(i, cap.isOpened())
    cap.release()
    