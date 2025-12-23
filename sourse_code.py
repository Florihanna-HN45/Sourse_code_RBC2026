import ssl
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import multiprocessing as mp
from multiprocessing import Queue
import serial


model = YOLO("fire_model.pt")
name = model.names

listCamera = ["rtsp://admin:ZDZIVU@192.168.2.1:554", "rtsp://admin:UCPRYR@192.168.2.2:554"]

def waringBuzzer(qbuzzer, COM):
    ss = serial.Serial(COM, baudrate=115200)
    while True:
        if not qbuzzer.empty():
            ss.write("1".encode(    ))
            qbuzzer.get_nowait()
        time.sleep(0.01)

def getResults(frame, model, thres):
    listOut = []
    rs = model(frame, stream=True)
    for r in rs:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(int(box.conf[0]*100))/100
            cls = int(box.cls[0])
            if conf > thres:
                listOut.append([cls, conf, [x1, y1, x2, y2]])
    return listOut

def warningOnFrame(frame, data):
    if len(data) == 0:
        return frame, False
    for cls, conf, [x1, y1, x2, y2] in data:
        label = name[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f"{label}, {conf*100}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return frame, True

def processImage(model, idCam, qbuzzer):
    cap = cv2.VideoCapture(idCam)
    thres = 0.5
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            listOut = getResults(frame, model, thres)
            frame, isWarning = warningOnFrame(frame, listOut)
            if isWarning:
                if qbuzzer.empty():
                    qbuzzer.put_nowait("1")
            cv2.imshow("cv2", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            print("Cannot get frame from Camera! Connect Again after 5 seconds:", idCam)
            cap = cv2.VideoCapture(idCam)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    qbuzzer = Queue(maxsize=1)
    # mp.Process(target=waringBuzzer, args=(qbuzzer, "COM8")).start()
    for idCam in listCamera:
        mp.Process(target=processImage, args=(model, idCam, qbuzzer)).start()
