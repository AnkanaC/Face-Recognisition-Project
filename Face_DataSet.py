# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:27:57 2023

@author: DELL
"""
import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3,640) #width
cam.set(4,480) #height
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id=input('\n Enter user id and press enter ')

print("\n [INFO] Initializing face capture....")

# Initialize individual sampling face count
count=0

while(True):
    ret, img = cam.read()    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30: #you can increase to 100-200 for more efficiency
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()