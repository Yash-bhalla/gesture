import cv2
import numpy as np
import os


GESTURE = "THANKYOU"   
SAVE_PATH = "dataset"
SEQ_LENGTH = 20


x1, y1, x2, y2 = 300, 100, 600, 400


gesture_path = os.path.join(SAVE_PATH, GESTURE)
os.makedirs(gesture_path, exist_ok=True)


sample_num = len(os.listdir(gesture_path))

cap = cv2.VideoCapture(0)

recording = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    hand = cv2.bitwise_and(roi, roi, mask=mask)

   
    hand = cv2.resize(hand, (64, 64))

  
    if recording:
        sample_folder = os.path.join(gesture_path, f"sample_{sample_num}")
        os.makedirs(sample_folder, exist_ok=True)

        cv2.imwrite(f"{sample_folder}/{frame_count}.jpg", hand)
        frame_count += 1

        cv2.putText(frame, f"Recording {frame_count}/{SEQ_LENGTH}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if frame_count >= SEQ_LENGTH:
            recording = False
            frame_count = 0
            sample_num += 1
            print("Saved one sample")

 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Hand", hand)

    key = cv2.waitKey(1)

    if key == ord('s'):
        recording = True

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()