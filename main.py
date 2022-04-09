import cv2
import time
import math
import numpy as np
from cvzone import HandTrackingModule as htm
import osascript

target_volume = 100
osascript.osascript("set volume output volume {}".format(target_volume))
result = osascript.osascript('get volume settings')
print(result)

cap = cv2.VideoCapture(0)
WIDTH,HEIGHT = 640,480
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
previous_time=0
detector = htm.HandDetector(detectionCon=0.9)
vol=100


while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        x1,y1 = lmList[4][0],lmList[4][1]
        x2,y2 = lmList[8][0],lmList[8][1]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        length=math.hypot(x2-x1,y2-y1)
        if length<20:
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        vol = np.interp(length,[20,200],[0,100])
        bar = np.interp(vol,[0,100],[400,150])
        osascript.osascript("set volume output volume {}".format(int(vol)))
        cv2.putText(img, f'VOL: {int(vol)}', (20, 100), cv2.FONT_ITALIC, 1, (0, 255, 255), 3)
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(bar)), (85,400), (0, 255, 0), cv2.FILLED)


    current_time= time.time()
    fps=1/(current_time-previous_time)
    previous_time=current_time
    cv2.putText(img,f'FPS: {int(fps)}',(20,50),cv2.FONT_ITALIC,1,(0,255,255),3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)

