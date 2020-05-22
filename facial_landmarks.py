# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def detect_face(img, shape_predictor_path = 'Data\shape_predictor_68_face_landmarks.dat'):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_path)

	# load the input image, resize it, and convert it to grayscale
	image = imutils.resize(img, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for i, rect in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		x, y, w, h = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	return image

def detect_facial_landmarks(img, shape_predictor_path = 'Data\shape_predictor_68_face_landmarks.dat'):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_path)

	img = imutils.resize(img, width=500)
	greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	rects = detector(greyed, 1)
	colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)]
	for i, rect in enumerate(rects):

		shape = predictor(greyed, rect)
		shape = face_utils.shape_to_np(shape)
		# print(shape.shape)
		for name, (i,j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			clone = img.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

			for (x,y) in shape[i:j]:
				cv2.circle(clone, (x,y), 1, (0,0,255), -1)
			
			(x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = img[y:y+h,x:x+w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)
			cv2.waitKey(0)
		
		out = face_utils.visualize_facial_landmarks(img, shape, colors=colors)
		cv2.imshow("Image", out)
		cv2.waitKey(0)



if __name__ == "__main__":
	
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=False,
		help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())

	img = cv2.imread(args["image"])
	# image = detect_face(img)
	# cv2.imshow('Frame', image)
	# cv2.waitKey(0)

	detect_facial_landmarks(img)
	