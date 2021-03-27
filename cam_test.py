import numpy as np
import cv2
import pandas as pd
from keras.models import load_model


model = load_model('model.h5')

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
nb=[]
while capture.isOpened():
    x, y, w, h = 0, 0, 300, 300
    ret, img = capture.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    thresh = thresh[y:y + h, x:x + w]
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            x, y, w, h = cv2.boundingRect(contour)
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            img1 = pd.DataFrame(newImage, dtype='float64')
            img2 = img1.values.reshape(-1, 28, 28, 1)
            img2 -= np.mean(img2, axis=1)
            nbr = model.predict(img2)
            nb = np.argmax(nbr)

    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(nb), (10, 380), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Contours", thresh)
    cv2.imshow("Results", img)
    k = cv2.waitKey(10)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
