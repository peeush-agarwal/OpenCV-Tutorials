---
layout: default
title: OpenCV Tutorials
---
# OpenCV Tutorials

Hands-on practice on tutorials from [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html)

+ How to capture video stream from IP Camera?
+ [Track colored object in a a video](colored_object_tracking.py)
+ [Thresholding](thresholding.py)
  + [Simple thresholding](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#simple-thresholding)
  + [Adaptive thresholding](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#adaptive-thresholding)
  + [Otsu's binarisation](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#otsus-binarization)

## Facial Landmarks using OpenCV, Dlib and imutils

[Code](facial_landmarks.py)

+ Detect facial landmarks from the given image
+ Extract ROI for Eyes, Nose, EyeBrows, JawLine, Mouth from image
+ Detect facial landmarks in real time using webcam

## Face detection using OpenCV DNN Caffe model

[Code](face_detector.py)

+ Detect face/s in given image path
+ Detect face/s in live webcam

## Comparison between using Non-Threaded vs Threaded processing of Webcam frames

[Code](threaded_webcam_processor_demo.py)

+ Use imutils to use WebcamVideoStream to read frames in a thread and process them in main thread
+ Use imutils to use FPS (Frames per second) calculation to compare the result.
