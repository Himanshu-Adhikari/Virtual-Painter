import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 80

folder = "header"
imagelist = os.listdir(folder)
overlayList = [cv2.imread(f'{folder}/{images}') for images in imagelist]
head = overlayList[0]
drawcolor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Function to convert OpenCV image to PIL format for Streamlit
def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# Display the initial canvas
canvas_img = st.image(cv2_to_pil(canvas), use_column_width=True, channels="BGR")

# Add an info button for instructions
st.sidebar.button("ℹ️ Instructions")

# Instructions to be displayed when the info button is clicked
instructions = """
**Instructions:**

1. Use two fingers (index and middle) to select color/eraser.
2. Use the index finger to draw.
3. 🤘 Use index finger and pinky to use the custom color.
"""

st.sidebar.markdown(instructions)

# Add a color picker for choosing brush color
default_color = "#%02x%02x%02x" % drawcolor
brush_color = st.sidebar.color_picker("Choose Brush Color", value=default_color)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmlist = detector.findPositions(img, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]

        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp = 0
            yp = 0
            if y1 < 62:
                if 30 < x1 < 120:
                    head = overlayList[0]
                    drawcolor=(0,0,255)
                elif 120 < x1 < 180:
                    head = overlayList[1]
                    drawcolor=(255,100,110)
                elif 200 < x1 < 300:
                    head = overlayList[2]
                    drawcolor=(0,255,255)
                elif 350 < x1 < 450:
                    head = overlayList[3]
                    drawcolor=(0,255,0)
                elif 500 < x1 < 600:
                    head = overlayList[4]
                    drawcolor=(0,0,0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawcolor, cv2.FILLED)

        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, drawcolor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, brushThickness)

            xp, yp = x1, y1
        if(fingers[1] and fingers[4]):
            drawcolor = tuple(int(brush_color[i:i + 2], 16) for i in (1, 3, 5))

            cv2.circle(img, (x1, y1), 10, drawcolor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    img[0:62, 0:640] = head

    # Update the Streamlit canvas with the new image
    canvas_img.image(cv2_to_pil(img))

# Close the webcam and destroy all OpenCV windows when the Streamlit app is closed
cap.release()
cv2.destroyAllWindows()
