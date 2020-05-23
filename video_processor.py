import cv2
import numpy as np
import facial_landmarks

def image_processing(img):
    # _, imgProc = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Webcam Processed', thresholded)
    imgProc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Greyed Video', imgProc)
    imgProc = cv2.Canny(imgProc, 60, 150)
    return imgProc

def read_video(cap):
    isFirst = True
    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break
        
        if isFirst:
            print('Press "q" to exit.\n\n')

            print(f'Frame size: {frame.shape}')

            isFirst = False

        cv2.imshow('Original Video', frame)
        imgProc = image_processing(frame)
        cv2.imshow('Processed Video', imgProc)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def video_facial_landmarks():
    fl = facial_landmarks.FacialLandmarks()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fl.detect_facial_landmarks_video(cap)
    cap.release()
    cv2.destroyAllWindows()
