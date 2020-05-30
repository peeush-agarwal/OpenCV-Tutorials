import datetime
import cv2
import imutils
from imutils.video import FPS, WebcamVideoStream
from threading import Thread

# class FPS:
#     def __init__(self):
#         self._start = None
#         self._end = None
#         self._n_frames = 0
    
#     def start(self):
#         self._start = datetime.datetime.now()
#         return self
    
#     def stop(self):
#         self._end = datetime.datetime.now()

#     def update(self):
#         self._n_frames += 1

#     def elapsed(self):
#         '''
#         Compute total seconds elapsed
#         '''
#         assert self._start is not None, "Call start() before calling elapsed()"
#         assert self._end is not None, "Call end() before calling elapsed()"
#         return (self._end - self._start).total_seconds()
    
#     def fps(self):
#         '''
#         Compute no. of frames per second
#         '''
#         try:
#             return self._n_frames / self.elapsed()
#         except ZeroDivisionError as ex:
#             print(self._start)
#             print(self._end)
#             print(self._n_frames)
#             return self._n_frames

# class WebcamVideoStream:
#     def __init__(self, source=0):
#         self.stream = cv2.VideoCapture(source, cv2.CAP_DSHOW)
#         assert self.stream is not None, "Couldn't read from Webcam source"
#         (self.has_frame, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         Thread(target=self.update, args=()).start()
#         return self
    
#     def update(self):
#         while True:
#             if self.stopped:
#                 return
            
#             (self.has_frame, self.frame) = self.stream.read()
    
#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True
#         self.stream.release()
#         cv2.destroyAllWindows()

if __name__ == "__main__":
    TOTAL_FRAMES = 120
    DISPLAY = True

    # Compare Non-Threaded vs Threaded reading from Webcam
    stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fps = FPS().start()

    while fps._numFrames < TOTAL_FRAMES:
        has_frame, frame = stream.read()
        frame = imutils.resize(frame, width=400)

        if DISPLAY:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            
        fps.update()
    
    fps.stop()
    stream.release()
    cv2.destroyAllWindows()

    print('*** Non-threaded results:***')
    print(f'Total seconds: {fps.elapsed():.2f} s')
    print(f'Total frames per second: {fps.fps()}')

    stream = WebcamVideoStream().start()
    fps = FPS().start()

    while fps._numFrames < TOTAL_FRAMES:
        frame = stream.read()
        frame = imutils.resize(frame, width=400)
        if DISPLAY:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
        fps.update()

    fps.stop()
    stream.stop()

    print('*** Threaded results:***')
    print(f'Total seconds: {fps.elapsed()} s')
    print(f'Total frames per second: {fps.fps()}')
