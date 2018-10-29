from threading import Thread, Lock, active_count
import threading
import cv2
import numpy as np
import imutils
import time
import queue
import os
import re

q = queue.Queue()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.draw = list()
        self.read_lock = Lock()
        self.detect_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread1 = Thread(target=self.update, args=())
        # self.thread1.daemon = True
        self.thread1.start()
        return self

    def update(self) :
        while self.started :
            # time.sleep(0.1)
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def forward_pass(self,frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        # detect_lock.acquire()
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                    # COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                # self.detect_lock.acquire()
                # numbers = 
                drawing = label, int(startX), int(startY), int(endX), int(endY), int(y), int(idx)
                # drawing.append(label)
                # drawing.append(map(int, numbers))
                # drawing = label, startX
                # self.draw = drawing
                q.put(drawing)
                # print(self.draw.shape)
                # self.detect_lock.release()
                # cv2.putText(frame, label, (startX, y),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            # cv2.imshow("Frame", frame)
            # # key = cv2.waitKey(1) & 0xFF
            # cv2.waitKey(1)
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break    
            # detect_lock.release()

        
    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def read_draw(self) :
        self.detect_lock.acquire()
        di =  self.draw
        self.detect_lock.release()
        return di

    def stop(self) :
        self.started = False
        self.thread1.join()
        detect_thread.join()


print("[INFO] loading model...")
start_time = time.time()
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
vs = WebcamVideoStream().start()
frame = vs.read()
detect_thread = Thread(target=vs.forward_pass, args=(frame,))
detect_thread.start()
detect_lock = Lock()
# detect_thread.join()
# print(threading.enumerate())

i, j, scale = 0, 0, 5
details = []
label, startX, startY, endX, endY, y, idx = [[] for i in range(7)]
while True :
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    if detect_thread.isAlive() == False:
        # print(detect_thread.isAlive())
    #     # detect_thread.kill()
        detect_thread = Thread(target=vs.forward_pass, args=(frame,))
        detect_thread.start()
    # details = vs.read_draw()
    if not q.empty():
        details = q.get()
        label, startX, startY, endX, endY, y, idx = details
        if len(label) != 0:
            break
while True :
    i += 1
    if i % scale == 0:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
    
    # if i % 100 == 0:
    #     print(active_count())
    #     print(threading.enumerate())

    if detect_thread.isAlive() == False:
        # print(detect_thread.isAlive())
    #     # detect_thread.kill()
        detect_thread = Thread(target=vs.forward_pass, args=(frame,))
        detect_thread.start()
    # details = vs.read_draw()
    
    if not q.empty():
        j += 1
        start = 1
        details = q.get()
        print(details)
        label, startX, startY, endX, endY, y, idx = details
        # for i in details:
        #     print(type(i), i)
        # print(label, startX, startY, endX, endY, y, idx)

    # print(details[0])
    # a,b,c,d,e = details
    # for i in details:
    #     print(i)
    # print(details.shape)
    # print(str(details[0]))
    # a = str(details)

    # all_ = (''.join((b)).split(','))
    # (startX, startY) = (''.join(b).split(','))[1]
            # (''.join(b).split(','))[0]
            # (''.join(b).split(','))[0]
            # (''.join(b).split(','))[0]
    # label, (startX, startY), (endX, endY), y, idx = (''.join(b).split(','))
    # print(a)
    # label = re.compile(r'\w+:\s\d+.\d+%')
    # print(label.search(a))
    # print(label)
    # print(a.strip('(').split(','))
    # print((b.split(',')[1]))



    # print((''.join(str(b)).split(','))[1])

    # for x,i in enumerate(details):
    #     if x == 0:
    #         label = i 
    #     elif x == 1:
    #         startX, startY = i
    #     elif x == 2:
    #         endX, endY = i
    #     elif x == 3:
    #         y = i
    #     elif x == 4:
    #         idx = i
    #     print((endX, endY))
        # print(x, i)

    # label, (startX, startY)
    # print(COLORS[idx])
    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
    # cv2.rectangle(frame, (details[1][0], details[1][1]), (details[2][0],details[2][1]),
    #             COLORS[details[4]], 2)
    # cv2.putText(frame, details[0], (details[1][0], details[3]),
    #     cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
    cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # detect_lock.release()
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27 :
        # vs.thread.join()
        time_now = time.time()
        print("FPS is ", (i/scale) / (time_now - start_time))
        print("refresh rate of object detection : ", j / (time_now - start_time))
        # vs.stop()
        break
# detect_lock.release()


vs.stop()
cv2.destroyAllWindows()
