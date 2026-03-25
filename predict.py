import cv2
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class GestureModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureModel, self).__init__()

        self.cnn = CNN()
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        features = self.cnn(x)

        features = features.view(B, T, -1)

        out, _ = self.gru(features)
        out = out[:, -1, :]

        return self.fc(out)


classes = ['COME', 'HELLO', 'HELP', 'NO', 'PLEASE', 'SORRY', 'STOP', 'THANKYOU', 'WATER', 'YES']

model = GestureModel(len(classes))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

cap = cv2.VideoCapture(0)

x1, y1, x2, y2 = 300, 100, 600, 400

SEQ_LENGTH = 10   
sequence = []

last_spoken = ""
CONF_THRESHOLD = 0.5


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

    
    hand_pixels = cv2.countNonZero(mask)
    HAND_THRESHOLD = 2000
    hand_present = hand_pixels > HAND_THRESHOLD

    hand = cv2.bitwise_and(roi, roi, mask=mask)
    hand = cv2.resize(hand, (64, 64))
    hand = hand / 255.0

    if hand_present:
        sequence.append(hand)

        if len(sequence) > SEQ_LENGTH:
            sequence.pop(0)

        
        if len(sequence) >= 8:
            input_data = np.array(sequence)
            input_data = np.transpose(input_data, (0, 3, 1, 2))
            input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(input_data)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            confidence = confidence.item()
            pred = pred.item()

            text = classes[pred]

           
            cv2.putText(frame, f"{text} ({confidence:.2f})",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3)

            
            if confidence > CONF_THRESHOLD and text != last_spoken:
                print("Speaking:", text)
                speak(text)
                last_spoken = text

    else:
        
        cv2.putText(frame, "NO HAND",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3)

        sequence.clear()
        last_spoken = ""

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Hand", hand)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()