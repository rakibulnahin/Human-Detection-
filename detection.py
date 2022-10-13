import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
video = cv2.VideoCapture(0)
while(True):
    _, frame = video.read()
    # frame =cv2.imread("r1.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(frame, 1.5, 3)
    print(bodies)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        print("body", bodies)
        # cv2.imshow("body detect", frame)

    cv2.imshow("body", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        break
cv2.destroyAllWindows()