"""Real-time webcam demo for the Sign Language Recognition model.

Opens the webcam, shows a square region of interest (ROI) in the centre of
the frame — hold your hand inside the box making an ASL letter sign — and
draws the model's predicted letter on screen, updating live.

Notes / honest limitations:
  - The model was trained on Sign Language MNIST: clean, centred, 28x28
    grayscale images with a dark background. Real webcam frames differ, so
    live accuracy will be lower than the test-set accuracy. Plain dark
    backgrounds and good lighting work best.
  - Only the 24 static letters are recognised (J and Z need motion).

Keys:  Q or ESC to quit.
Usage: python webcam_demo.py
"""
import os
import sys

import cv2
import joblib
import numpy as np

from common import LABEL_TO_LETTER

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best_model.joblib")
NAME_PATH = os.path.join(HERE, "models", "best_model_name.txt")

ROI_SIZE = 220          # on-screen box size in pixels
IMG = 28                # model input size


def preprocess(roi_bgr: np.ndarray) -> np.ndarray:
    """Convert an ROI frame to the model's input format: 1x784 floats in [0,1]."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    small = cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(1, -1)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit("Model not found — run train.py first.")
    model = joblib.load(MODEL_PATH)
    name = open(NAME_PATH).read().strip() if os.path.exists(NAME_PATH) else "model"
    has_proba = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam (no camera available on this machine).")
    print(f"Webcam demo running with model: {name}. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]
        x1, y1 = (w - ROI_SIZE) // 2, (h - ROI_SIZE) // 2
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE
        roi = frame[y1:y2, x1:x2]

        pred_label = int(model.predict(preprocess(roi))[0])
        letter = LABEL_TO_LETTER[pred_label]
        text = f"Letter: {letter}"
        if has_proba:
            conf = float(np.max(model.predict_proba(preprocess(roi))))
            text += f"  ({conf * 100:.0f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.putText(frame, text, (14, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, "Hold your hand inside the box - Q to quit",
                    (x1, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Sign Language Recognition - webcam demo", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
