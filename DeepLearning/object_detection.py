'''
Inspired by PyImageSearch blog: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
This python script demonstrates:
- How to load pre-trained Caffe model into OpenCV?
- Use this model to detect and localize objects in image using OpenCV.
'''
import argparse
import numpy as np
import cv2

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to input image", required=True)
ap.add_argument("-p", "--prototxt", help="Path to caffe model's prototxt", required=True)
ap.add_argument("-m", "--model", help="Path to caffe model's weight file", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize class labels and bounding box colors
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
"car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"potted_plant", "sheep", "sofa", "train", "tv_monitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load pre-trained model
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

# Forward pass blob through the network and obtain the detections and predictions
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e. probability) associated with the predictions
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is greater than min confidence
    if confidence > args["confidence"]:
        # Extract the index of the class label from the detections, 
        # then compute the (x,y) co-ordinates of the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Image", image)
cv2.waitKey(0)