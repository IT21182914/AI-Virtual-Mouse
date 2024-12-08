import cv2
import numpy as np
from pynput.mouse import Controller, Button
from HandTrackingModule import handDetector

# Webcam dimensions
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

# Initialize variables
pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

# Mouse Controller
mouse = Controller()

# Webcam and Hand Detector
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(maxHands=1)

while True:
    # 1. Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 2. Get the tip of the index and middle fingers
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, 1920))  # Adjust screen size
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, 1080))

            # Smoothen values for better control
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            # Move the mouse
            mouse.position = (cLocX, cLocY)
            pLocX, pLocY = cLocX, cLocY

        # 5. Both Index and Middle Fingers: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length = np.hypot(x2 - x1, y2 - y1)

            # Click if distance is short
            if length < 40:
                mouse.click(Button.left, 1)

    # 6. Display
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
