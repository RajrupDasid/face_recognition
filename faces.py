import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


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
        img_item="people-face.png"
        cv2.imwrite(img_item,roi_gray)
        color=(255,0,0) #BGR way for histogram calculation and meshing
    #display these on resulting frame color scale
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
#trigger success capture release 
cap.release()
cv2.destroyAllWindows()

