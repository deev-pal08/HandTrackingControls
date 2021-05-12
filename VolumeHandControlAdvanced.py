import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0
detector = htm.handDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
volColor = (255, 0, 0)
while True:
    success, img = cap.read()
    #Find Hand
    img = detector.findHands(img)
    lmList, bBox = detector.findPositions(img, draw=True)
    if len(lmList) != 0:
        # Filter based on size
        area = (bBox[2] - bBox[0]) * (bBox[3] - bBox[1]) // 100
        if 250 < area < 1500:
            # Find distance between index and thumb
            length, img, lineInfo = detector.findDistance(4, 8 , img)

            # Convert Volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()

            # If small finger is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                volColor = (0, 255, 0)
            else:
                volColor = (255, 0, 0)

    # Drawings
    cv2.rectangle(img, (50,150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Volume: {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, volColor, 3)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
