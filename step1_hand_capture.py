import cv2
import numpy as np

cap = cv2.VideoCapture(0)


x1, y1, x2, y2 = 300, 100, 600, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = frame[y1:y2, x1:x2]

    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

 
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    
    result = cv2.bitwise_and(roi, roi, mask=mask)

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Hand", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()