import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils

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

def video_facial_landmarks(cap, shape_predictor_path = 'Data\shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break
        
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for i, rect in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x,y) in shape:
                cv2.circle(frame, (x,y), 1, (0,0,255),-1)
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
