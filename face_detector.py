from pathlib import Path
from cv2 import resize
from cv2.dnn import readNetFromCaffe, blobFromImage
import numpy as np

class FaceDetector:
    """ Face Detector class
    """
    def __init__(self, prototype: Path=None, model: Path=None,
                 confidenceThreshold: float=0.6):
        self.prototype = prototype
        self.model = model
        self.confidenceThreshold = confidenceThreshold
        if self.prototype is None:
            raise Exception("must specify prototype '.prototxt.txt' file "
                                        "path")
        if self.model is None:
            raise Exception("must specify model '.caffemodel' file path")
        self.classifier = readNetFromCaffe(str(prototype), str(model))
    
    def detect(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces_with_confidence = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces_with_confidence.append((np.array([startX, startY, endX-startX, endY-startY]), confidence))
        return faces_with_confidence

def show_faces_frame(face_detector, img):
    import cv2
    rects_and_confidence = face_detector.detect(img)

    for rect, confidence in rects_and_confidence:
        x, y, w, h = rect
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f'C:{confidence*100.0:.2f}%', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,0,255), 2)
        # print(f'{confidence:.2f} %')
    
    cv2.imshow('Faces', img)

def visualize_faces(face_detector, img_path):
    import cv2
    import imutils

    img = cv2.imread(img_path)
    img = imutils.resize(img, width=500)
    show_faces_frame(face_detector, img)
    cv2.waitKey(0)

def visualize_faces_webcam():
    import cv2
    import imutils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        
        show_faces_frame(face_detector, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PROTO_PATH = 'Assets\\Face_Detector_Model\\deploy.prototxt.txt'
    MODEL_PATH = 'Assets\\Face_Detector_Model\\res10_300x300_ssd_iter_140000.caffemodel'
    IMAGES_PATH = 'Data\\Images'
    
    face_detector = FaceDetector(prototype=PROTO_PATH, model=MODEL_PATH)

    import os

    for img_path in os.scandir(IMAGES_PATH):
        visualize_faces(face_detector, img_path.path)
    
    visualize_faces_webcam()
        