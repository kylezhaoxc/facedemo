import cv2
from facereco import face_reco
import datetime
import checkSum
from multiprocessing.pool import ThreadPool


pool = ThreadPool(processes=4)

rootdir = "/home/pi/refImg"
blackListDir = "/home/pi/blackList"
handler = face_reco()
handler.init_with_images(rootdir)
blacklist_handler = face_reco()
blacklist_handler.init_with_images(blackListDir)

video_capture = cv2.VideoCapture(0)
videoIndex = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter("cam0"+str(videoIndex)+".avi",fourcc,20.0,(640,480))
starttime = datetime.datetime.now()
NoFaceCountDown = -10
MatchBlackList = False
process_this_frame = 10
font = cv2.FONT_HERSHEY_DUPLEX
faceCount=0
top=0
bottom =0
left =0
right=0
name = ""
def handleframe(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    locs = []
    names=[]
    
    # Only process every other frame of video to save time
    global process_this_frame
    global faceCount
    #print(process_this_frame)
    if process_this_frame==0:
        faceCount=0
        blacklist = blacklist_handler.process_one_pic(rgb_small_frame)
        global MatchBlackList
        if(blacklist[2]==True):
            MatchBlackList=True
            locs = blacklist[0]
            names=blacklist[1]
        else:
            MatchBlackList=False
            result = handler.process_one_pic(rgb_small_frame)
            locs = result[0]
            names=result[1]
        process_this_frame=20
        #print("reset!")
    else:
        process_this_frame = process_this_frame-1
        return frame
    # Display the results
    global top
    global right 
    global bottom
    global left
    global name
    for (top, right, bottom, left), name in zip(locs, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        faceCount=faceCount+1
        # Draw a label with a name below the face
    if(MatchBlackList==True):
        cv2.putText(frame, "Black List", (0, 30), font, 1.5, (0, 0, 0), 2)      
    else:
        if(faceCount==0):
            global NoFaceCountDown
            if(NoFaceCountDown>0):
                cv2.putText(frame, "Shaded", (0, 30), font, 1.5, (0, 255, 255), 2)
                NoFaceCountDown=NoFaceCountDown-1
            else:
                cv2.putText(frame, "Not Started", (0, 30), font, 1.5, (0, 0, 255), 2)
        else:
            NoFaceCountDown=50
    return frame
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    async_result= pool.apply_async(handleframe,(frame,))
    outframe = async_result.get()
    if(MatchBlackList==False):
        cv2.rectangle(outframe, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(outframe, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(outframe, str(name), (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
    else:
        cv2.rectangle(outframe, (left, top), (right, bottom), (255, 255, 255), 2)
        cv2.rectangle(outframe, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
        cv2.putText(outframe, str(name), (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
    # Display the resulting image
    cv2.imshow('Video', outframe)

    out.write(outframe)
    curr = datetime.datetime.now()
    if (curr-starttime).total_seconds()>3600:
        starttime=curr
        out.release()
        videoIndex = videoIndex+1
        out=cv2.VideoWriter("cam0"+str(videoIndex)+".avi",fourcc,20.0,(640,480))
    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()