import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        left_eye = leye.detectMultiScale(roi_gray)
        right_eye = reye.detectMultiScale(roi_gray)

        for (lx, ly, lw, lh) in left_eye:
            l_eye = roi_color[ly:ly + lh, lx:lx + lw]
            count += 1
            l_eye_gray = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye_gray = cv2.resize(l_eye_gray, (24, 24))
            l_eye_gray = l_eye_gray / 255
            l_eye_gray = np.reshape(l_eye_gray, (1, 24, 24, 1))
            lpred = model.predict(l_eye_gray)
            if lpred[0][0] < 0.5:
                lbl = 'Open'
            else:
                lbl = 'Closed'
            break

        for (rx, ry, rw, rh) in right_eye:
            r_eye = roi_color[ry:ry + rh, rx:rx + rw]
            count += 1
            r_eye_gray = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye_gray = cv2.resize(r_eye_gray, (24, 24))
            r_eye_gray = r_eye_gray / 255
            r_eye_gray = np.reshape(r_eye_gray, (1, 24, 24, 1))
            rpred = model.predict(r_eye_gray)
            if rpred[0][0] < 0.5:
                lbl = 'Open'
            else:
                lbl = 'Closed'
            break

    if lbl == 'Closed':
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
