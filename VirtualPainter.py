import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (255, 0, 255)
brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  #Index Finger Tip Landmark
        x2, y2 = lmList[12][1:]  # Middle Finger Tip Landmark
        # 3. Check Which Fingers Are Up
        fingers = detector.fingersUp()
        # 4. Selection mode : Two Fingers Are Up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #Checking for Click
            if y1 < 125:
                if 80 < x1 < 180:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 230 < x1 < 330:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 380 < x1 < 480:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 530 < x1 < 640:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. Drawing Mode : Index Finger Are Up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                brushThickness = 50
            else:
                brushThickness = 15
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # Setting the header Image
    img[0:125, 0:640] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
