from threading import Thread, Lock
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import time

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    print("[INFO] loading model...")
    start = time.time()
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    vs = WebcamVideoStream().start()
    i = 0
    while True :
        i += 1
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        """
        """
        # (h, w) = frame.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        #     0.007843, (300, 300), 127.5)

        # # pass the blob through the network and obtain the detections and
        # # predictions
        # net.setInput(blob)
        # detections = net.forward()



        """
        """
        cv2.putText(frame, str(i), (100,100),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 :
            print("FPS is ", i / (time.time() - start))
            break
    
    vs.stop()
    cv2.destroyAllWindows()