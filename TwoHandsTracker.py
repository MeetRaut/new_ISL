import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 30  # Increase the offset for a wider frame
imgSize = 350  # Increase the imgSize for a larger image

folder = "Data/R"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if len(hands) == 2:  # Ensure that both hands are detected
        # Get the bounding boxes of both hands
        bbox1 = hands[0]['bbox']
        bbox2 = hands[1]['bbox']

        # Combine the bounding boxes to get a wider frame
        x_min = min(bbox1[0], bbox2[0]) - offset
        y_min = min(bbox1[1], bbox2[1]) - offset
        x_max = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]) + offset
        y_max = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]) + offset

        imgCrop = img[y_min:y_max, x_min:x_max]

        # Resize the frame
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

        imgWhite[:imgSize, :imgSize] = imgCrop

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord('0'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    
    elif key == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()