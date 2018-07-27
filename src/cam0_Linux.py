import cv2
from facereco_linux import face_reco
import datetime
import serial
import time
import threading
import queue
import os
import face_recognition
from flask import Flask,request,make_response,Response
from flask_cors import CORS
import json
import numpy as np
import socket

imgqueue = queue.Queue()
lastopentime=datetime.datetime.now()
def send_open_command():
    opencommand = b'\x55\xff\xff\x01\x00\x01\x01\x56\x88'
    curr = datetime.datetime.now()
    global lastopentime
    if((curr-lastopentime).total_seconds()>15):    
        time.sleep(1)
        ser = serial.Serial('/dev/ttyUSB0',9600,timeout=1)
        ser.write(opencommand)
        lastopentime = curr
    
rootdir = "/home/hd/refImg"
handler = face_reco()
handler.init_with_images(rootdir)

video_capture = cv2.VideoCapture(0)
videoIndex = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter("cam0"+str(videoIndex)+".avi",fourcc,20.0,(640,480))
starttime = datetime.datetime.now()
NoFaceCountDown = -10
MatchBlackList = False
process_this_frame = True
font = cv2.FONT_HERSHEY_DUPLEX
locs = []
names = []

def SaveVideoFromQueue():
    global out,starttime,videoIndex
    while True:
        if(imgqueue.empty()):
            continue
        frame = imgqueue.get_nowait()
        out.write(frame)
        curr = datetime.datetime.now()
        if (curr-starttime).total_seconds()>3600:
            starttime=curr
            out.release()
            videoIndex = videoIndex+1
            out=cv2.VideoWriter("cam0"+str(videoIndex)+".avi",fourcc,20.0,(640,480))
videoThread = threading.Thread(target=SaveVideoFromQueue)
videoThread.start()

app=Flask(__name__)
CORS(app)
@app.route('/updateFeature',methods=['POST'])
def getFeature():
    global handler
    metadata = str(request.data,encoding="utf-8")
    #print(metadata)
    meta = json.loads(metadata)
    raw = json.loads(meta['metadataString'])
    imgtype= meta['imgtype']
    raw['face-data']=np.array(raw['face-data'])

    if(imgtype=='blackList'):
        handler.blackListNames.append(raw['name'])
        handler.blackListNames.append(raw['name'])
    else:
        handler.knownNames.append(raw['name'])
    handler.knownFaces.append(raw['face-data'])
    handler.Save()

    return Response('ok')   
def StartFlask():
    global app
    app.run(host='0.0.0.0',port=5001,debug = False)
flaskThread = threading.Thread(target=StartFlask)
flaskThread.start()

def CVReco(rgb_small_frame):
    global faceCount,handler,MatchBlackList,locs,names
    faceCount=0
    result = handler.process_one_pic(rgb_small_frame)
    if(result[2]==True):
        MatchBlackList=True
    else:
        MatchBlackList=False
    locs = result[0]
    names=result[1]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        cvthread = threading.Thread(target = CVReco,args = (rgb_small_frame,))
        cvthread.start()
    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(locs, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        if(MatchBlackList==False):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
            if(str(name)!='Unknown'):
                thread = threading.Thread(target = send_open_command)
                thread.start()
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
        faceCount=faceCount+1
        # Draw a label with a name below the face
    if(MatchBlackList==True):
        cv2.putText(frame, "Black List", (0, 30), font, 1.5, (0, 0, 0), 2)      
    else:
        if(faceCount==0):
            if(NoFaceCountDown>0):
                cv2.putText(frame, "Shaded", (0, 30), font, 1.5, (0, 255, 255), 2)
                NoFaceCountDown=NoFaceCountDown-1
            else:
                cv2.putText(frame, "Not Started", (0, 30), font, 1.5, (0, 0, 255), 2)
        else:
            NoFaceCountDown=50
    # Display the resulting image
    cv2.imshow('Video', frame)
    imgqueue.put_nowait(frame)
    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        os._exit()

