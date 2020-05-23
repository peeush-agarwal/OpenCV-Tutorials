import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np

class FacialLandmarks(object):
    def __init__(self, shape_predictor_path = 'Data\shape_predictor_68_face_landmarks.dat'):
        self.__detector__ = dlib.get_frontal_face_detector()
        self.__predictor__ = dlib.shape_predictor(shape_predictor_path)
    
    def detect_facial_landmarks(self, img_path):
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = self.__detector__(gray, 1)

        for i, rect in enumerate(rects):
            shape = self.__predictor__(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f'Face:{i+1}', (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

            for (x,y) in shape:
                cv2.circle(img, (x,y), 1, (0,0,255), -1)
        
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def extract_facial_landmarks(self, img_path):
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = self.__detector__(gray, 1)

        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
        (168, 100, 168), (158, 163, 32),
        (163, 38, 32), (180, 42, 220), (255, 0,255 )]
        for i, rect in enumerate(rects):
            shape = self.__predictor__(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for name, (i,j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                clone = img.copy()
                cv2.putText(clone, name, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                for (x,y) in shape[i:j]:
                    cv2.circle(clone, (x,y), 1, (0,0,255), -1)

                (x,y,w,h) = cv2.boundingRect(np.array(shape[i:j]))
                roi = img[y:y+h,x:x+w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                cv2.imshow('Image', clone)
                cv2.imshow('ROI', roi)
                cv2.waitKey(0)

            out = face_utils.visualize_facial_landmarks(img, shape, colors=colors)
            cv2.imshow('Output', out)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_facial_landmarks_video(self, cap):
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.__detector__(gray, 1)

            for i, rect in enumerate(rects):
                shape = self.__predictor__(gray, rect)
                shape = face_utils.shape_to_np(shape)

                (x,y,w,h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f'Face:{i+1}', (x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                for (x,y) in shape:
                    cv2.circle(frame, (x,y), 1, (0,0,255), -1)
            
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    

if __name__ == "__main__":
    facial_landmarks = FacialLandmarks()

    # Video processing
    cap = cv2.VideoCapture(0)
    facial_landmarks.detect_facial_landmarks_video(cap)
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

    # Extract facial landmarks from image
    facial_landmarks.extract_facial_landmarks('Data\Images\example_01.jpg')

    # Detect facial landmarks from image
    facial_landmarks.detect_facial_landmarks('Data\Images\example_01.jpg')
    facial_landmarks.detect_facial_landmarks('Data\Images\example_02.jpg')
    facial_landmarks.detect_facial_landmarks('Data\Images\example_03.jpg')