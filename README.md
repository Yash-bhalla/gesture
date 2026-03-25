# Hand Gesture Recognition System

A real-time hand gesture recognition system that detects sign-language-inspired gestures via webcam and speaks them aloud using text-to-speech. Built with PyTorch (CNN + GRU), OpenCV, and pyttsx3.

---

## Supported Gestures

| Gesture | Description |
|---------|-------------|
| **Hello** | Hello wave |
| **Yes** | Closed fist, up-down motion |
| **No** | One finger wave |
| **Water** | Two-finger tap-tap |
| **Come** | Lateral half-hand, forward-downward motion |
| **Sorry** | Closed fist diagonal (top-left to bottom-right) |
| **Please** | Open fist, small clockwise motion |
| **Help** | Open fist, upward and downward |
| **Stop** | Open fist, backward and forward |
| **Thank You** | Open fist sideways, open-close-open |

---

## Project Structure

```
gesture-recognition/
├── step1_hand_capture.py   # Test & tune hand detection
├── step2_collect_data.py   # Record gesture samples
├── train.py                # Train the CNN+GRU model
├── predict.py              # Real-time gesture prediction
├── model.pth               # Saved trained model weights
├── gesture.txt             # Gesture descriptions reference
└── dataset/                # Collected training data (created by step 2)
    ├── HELLO/
    │   ├── sample_0/
    │   │   ├── 0.jpg
    │   │   └── ...
    │   └── ...
    └── ...
```

---

## Requirements

- Python 3.8+
- Webcam

### Python Dependencies

```bash
pip install torch torchvision opencv-python numpy pyttsx3
```

---

## How to Use

### Step 1 — Test Hand Detection

Run this first to verify your webcam works and tune the skin-color detection for your lighting conditions:

```bash
python step1_hand_capture.py
```

A window will open showing three views: the raw frame, the HSV mask, and the detected hand region. Adjust the HSV thresholds in the script if your skin tone is not detected well.

Press **Q** to quit.

---

### Step 2 — Collect Training Data

Set the gesture name you want to record at the top of the file:

```python
GESTURE = "HELLO"   # Change this for each gesture
```

Then run:

```bash
python step2_collect_data.py
```

- Press **S** to start recording one sample (records 20 frames automatically).
- Repeat to collect multiple samples per gesture.
- Collect at least **30–50 samples** per gesture for good accuracy.
- Change `GESTURE` and repeat for all 10 gestures.

Press **Q** to quit.

---

### Step 3 — Train the Model

Once data is collected for all gestures, run:

```bash
python train.py
```

This trains a CNN+GRU model for 10 epochs and saves the weights to `model.pth`. Training progress (loss and accuracy) is printed each epoch.

---

### Step 4 — Run Real-Time Prediction

```bash
python predict.py
```

Place your hand inside the green rectangle on screen. The detected gesture and confidence score are shown on screen, and the gesture is spoken aloud when confidence exceeds 50%.

Press **Q** to quit.

---

## Model Architecture

The model uses a two-stage architecture to capture both spatial and temporal features from gesture sequences:

```
Input: Sequence of 10 frames (64×64 RGB images)
        ↓
CNN (per frame):
  Conv2d(3→16) → ReLU → MaxPool
  Conv2d(16→32) → ReLU → MaxPool
  Conv2d(32→64) → ReLU → MaxPool
  Linear(4096→128)
        ↓
GRU (across frames):
  GRU(128→64, sequence)
  Take last hidden state
        ↓
Fully Connected → 10 classes
```

| Component | Details |
|-----------|---------|
| CNN output | 128-dim feature vector per frame |
| GRU hidden size | 64 |
| Sequence length | 10 frames (prediction), 20 frames (training) |
| Input image size | 64×64 pixels |
| Number of classes | 10 |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| Epochs | 10 |

---

## Hand Detection

The system uses HSV color-space skin detection to isolate the hand region:

- **ROI (Region of Interest):** A fixed 300×300 pixel box displayed on screen
- **HSV Range:** Hue 0–25, Saturation 20–255, Value 70–255
- **Morphological cleanup:** Opening + dilation to remove noise
- **Hand presence threshold:** Detected if >2000 skin-colored pixels are found in the ROI

> **Note:** Detection works best under consistent, well-lit conditions. Avoid backgrounds with skin-like colors.

---

## Tips for Better Accuracy

- **Lighting:** Use consistent, diffuse lighting with no harsh shadows.
- **Background:** Use a plain, non-skin-colored background.
- **Data variety:** Record samples at different speeds and slight angle variations.
- **More data:** More samples per gesture generally leads to higher accuracy.
- **Skin tone calibration:** Adjust HSV thresholds in `step1_hand_capture.py` if the mask does not capture your hand well.

---

