import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/1"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    for hand in hands:
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCalc = math.ceil((k * w))
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            wGap = math.ceil((imgSize - wCalc) / 2)
            imgWhite[:, wGap:wCalc + wGap] = imgResize

        else:
            k = imgSize / w
            hCalc = math.ceil((k * h))
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap: hGap + hCalc, :] = imgResize

        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 2)

    cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('0'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    
    if key == ord('q'):
        break  # exit the loop if 'q' is pressed

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
