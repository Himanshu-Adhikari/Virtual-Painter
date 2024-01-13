import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


brushThickness=15
eraserThickness=80


folder = "header"
# get all files in the list
imagelist = os.listdir(folder)
# print(imagelist)
# store all images in a list (0-255) as matrices
overlayList = []
for images in imagelist:
    image = cv2.imread(f'{folder}/{images}')
    overlayList.append(image)
# print(len(overlayList))
head = overlayList[0]
drawcolor=(0,0,255)
# print(head.shape)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector=htm.handDetector(detectionCon=0.85)

xp,yp=0,0

canvas=np.zeros((480, 640, 3), dtype=np.uint8)

while 1:
    #1.import image
    success, img = cap.read()
    img=cv2.flip(img,1)

    #2. Find hand landmarks
    img=detector.findHands(img,draw=False)
    lmlist=detector.findPositions(img,draw=False)

    if len(lmlist)!=0:
        # print(lmlist)

        #tip of index and middle finger
        x1,y1=lmlist[8][1],lmlist[8][2]
        x2,y2=lmlist[12][1],lmlist[12][2]



        #3. Check finger is up or not
        fingers=detector.fingersUp()
        # print(fingers)

        #4. If selection mode-- 2fingers up
        if(fingers[1] and fingers[2]):
            # print("selection mode")
            xp=0
            yp=0
            if(y1<62):
                # print("banner")
                if (30<x1<120):
                    head=overlayList[0]
                    drawcolor=(0,0,255)
                elif(120<x1<180):
                    head=overlayList[1]
                    drawcolor=(255,100,110)
                elif(200<x1<300):
                    head=overlayList[2]
                    drawcolor=(0,255,255)
                elif(350<x1<450):
                    head=overlayList[3]
                    drawcolor=(0,255,0)
                elif(500<x1<600):
                    head=overlayList[4]
                    drawcolor=(0,0,0)
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawcolor,cv2.FILLED)
        #5. If drawing mode-- index finger up
        if(fingers[1] and not fingers[2]):
            cv2.circle(img,(x1,y1),10,drawcolor,cv2.FILLED)
            # print("drawing mode")
            if(xp==0 and yp==0):
                xp,yp=x1,y1

            if(drawcolor==(0,0,0)):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, brushThickness)

            xp,yp=x1,y1

    imgGray=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)

    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,canvas)


    #applying the image as header
    img[0:62, 0:640] = head
    # img=cv2.addWeighted(img,0.5,canvas,0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("canvas",canvas)
    cv2.waitKey(1)
