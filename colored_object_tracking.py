'''
In HSV, it is more easier to represent a color than RGB color-space.
Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
'''

import numpy as np
import cv2
import argparse
from collections import deque

lower_blue_hsv = np.array([110, 50, 50])
upper_blue_hsv = np.array([130, 255, 255])

pts = deque(maxlen=64)

def bgr_color_value(color='blue'):
    if color == 'blue':
        return np.uint8([[[255,0,0 ]]])
    if color == 'green':
        return np.uint8([[[0,255,0 ]]])
    if color == 'red':
        return np.uint8([[[0,0,255 ]]])
    raise NotImplementedError(f'{color} is not implemented yet')
def bgr_to_hsv(color='blue'):
    if isinstance(color, str):
        bgr_value = bgr_color_value(color)
    elif isinstance(color, np.ndarray):
        bgr_value = color
    else:
        print(type(color))
        raise ValueError(color)

    hsv_value = cv2.cvtColor(bgr_value, cv2.COLOR_BGR2HSV)
    return hsv_value
def hsv_limits(hsv_value):
    h, _, _ = hsv_value.squeeze((0,1))
    lower_limit = np.array([h-10,100, 100])
    upper_limit = np.array([h+10,255, 255])
    return lower_limit, upper_limit

class HSV_Limits(object):
    def __init__(self, color = 'blue'):
        '''
        color = 'blue' or 'green' or 'red' or (1, 1, 3) np.ndarray
        '''
        self.color = color
        hsv_value = bgr_to_hsv(self.color)
        limits = hsv_limits(hsv_value)
        self.lower, self.upper = limits

def detect_object(img, color_limits, display_masks = False):
    # Change from BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    final_mask = None
    for i, cl in enumerate(color_limits):
        # print(cl.color, cl.lower, cl.upper)
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, cl.lower, cl.upper)

        if display_masks:
            cv2.imshow(f'Mask:{cl.color}', mask)
        
        if final_mask is not None:
            final_mask = cv2.bitwise_or(final_mask, mask)
        else:
            final_mask = mask

    if display_masks:
        cv2.imshow('Final Mask', final_mask)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=final_mask)
    
    return final_mask, res

def track_objects(cap, color_limits, display_masks=False):
    while cap.isOpened():
        # Take each frame
        _, frame = cap.read()

        # Break from loop when frame is not available
        if frame is None:
            break

        mask, res = detect_object(frame, color_limits, display_masks)
        
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
    
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 5:
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
                pts.appendleft(center)
                for i in range (1,len(pts)):
                    if pts[i-1]is None or pts[i] is None:
                        continue
                    thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i-1],pts[i],(0,0,225),thick)

        # Show images
        cv2.imshow('Original', frame)
        cv2.imshow('Result', res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
