import cv2
import argparse
from video_processor import read_video
from colored_object_tracking import track_objects, HSV_Limits
from thresholding import apply_thresholding
from facial_landmarks import detect_face

ap = argparse.ArgumentParser()
ap.add_argument('-wc', '--web_cam', default=False, action='store_true', help="Use Integrated webcam")
ap.add_argument('-u', '--url', help="Url for VideoStream if not from Integrated webcam.")
args = ap.parse_args()

if __name__ == "__main__":
    if args.web_cam:
        print('Processing video from webcam')
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) # (0) not working.
    elif args.url:
        print(f'Processing video from {args.url}')
        # url='http://192.168.1.40:8080/video'
        cap = cv2.VideoCapture(args.url)
    else:
        ap.print_help()
        exit(0)

    # read_video(cap)
    
    # color_limits = list()
    # color_limits.append(HSV_Limits())
    # color_limits.append(HSV_Limits(color='green'))
    # color_limits.append(HSV_Limits(color='red'))

    # track_objects(cap, color_limits, display_masks=True)

    # apply_thresholding(cap)

    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        # print(ret)

        if frame is None:
            break

        img = detect_face(frame)
        
        cv2.imshow('Processed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()