import numpy as np
import cv2
import pickle

recognizer=cv2.face.LBPHFaceRecognizer_create()
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer.read("recognizer.yml")
labels={}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)
while True:
    #capture thre images frame by frame
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5) # number can be bumped up for accuracy
    for (x,y,w,h) in faces: # faces value for regeion of interests | caution not work for faces wearing glasses
        print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w] #regeion of interest of gray scale
        roi_color=frame[y:y+h, x:x+w] # regeion of interest for color scale
        # recognizer 
        id_, conf=recognizer.predict(roi_gray)
        if conf >=45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        #extras
        img_item="people-face.png"
        cv2.imwrite(img_item,roi_gray)
        # square up around faces 
        color=(255,0,0) #BGR way for histogram calculation and meshing
        stroke=2
        end_cord_x= x + w
        end_cord_y= y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    #display these on resulting frame color scale
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
#trigger success capture release 
cap.release()
cv2.destroyAllWindows()

