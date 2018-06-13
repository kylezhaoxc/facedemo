import cv2
from facereco import face_reco

rootdir = "D:\\facedemo\\imgs"
handler = face_reco()
handler.init_with_images(rootdir)

video_capture = cv2.VideoCapture(0)
NoFaceCountDown = -10
process_this_frame = True
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        faceCount=0
        result = handler.process_one_pic(rgb_small_frame)
        locs = result[0]
        names=result[1]
    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(locs, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        faceCount=faceCount+1
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
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
    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break