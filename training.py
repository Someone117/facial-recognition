# tries to recognize a face from a webcam
from sklearn.svm import LinearSVC, LinearSVR
import argparse
import os
import math
from PIL import Image
import mtcnn
import matplotlib.pyplot as mpl
import cv2
import time
from imutils import paths


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)


def detect_bounding_box(vid, i=0):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cropped_face = gray_image[y:y+h, x:x+w]
        # resize image to 200 by 200
        resized_face = cv2.resize(cropped_face, (200, 200)) # use for ai
        cv2.imshow("Face", resized_face)
        cv2.imwrite(str(i) + ".jpg", resized_face)
i=0
while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    
    detect_bounding_box(
        video_frame, i
    )  # apply the function we created to the video frame
    i+=1
    time.sleep(0.1)
    if i > 100:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()