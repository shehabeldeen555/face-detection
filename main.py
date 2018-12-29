import cv2 as cv
import numpy as np
import tensorflow as tf

names = ['ashraf', 'joseph', 'magdy', 'ref', 'sayed', 'Shehab']
model = tf.keras.models.load_model('best-conv.h5')

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x = x - 30
        y = y - 50
        w = w + 60
        h = h + 60
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y:y + h, x:x + w]
        face = cv.resize(face, (48, 48))
        face_to_predict = np.zeros(shape=(1, 48, 48, 3))
        face_to_predict[0] = face
        name = names[np.argmax(model.predict(face_to_predict))]
        cv.putText(img, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 125, 255), 2, cv.LINE_AA)

    # img = cv.flip(img, 1)
    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
