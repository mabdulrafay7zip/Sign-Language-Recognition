"""Run the saved best model on held-out test images (no webcam needed).

Loads models/best_model.joblib, predicts the first N images of the official
test set, prints true vs predicted letters, and saves a labelled grid to
results/prediction_demo.png. Proves the full inference path works end to end.

Usage:  python predict_test.py [N]     (default N=16)
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_data, to_letter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best_model.joblib")
NAME_PATH = os.path.join(HERE, "models", "best_model_name.txt")
OUT = os.path.join(HERE, "results", "prediction_demo.png")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if not os.path.exists(MODEL_PATH):
        sys.exit("Model not found — run train.py first.")
    model = joblib.load(MODEL_PATH)
    name = open(NAME_PATH).read().strip() if os.path.exists(NAME_PATH) else "model"
    _, _, X_test, y_test = load_data()

    X, y = X_test[:n], y_test[:n]
    pred = model.predict(X)
    correct = int((pred == y).sum())
    print(f"Model: {name}")
    print(f"Predicting {n} held-out test images...\n")
    for i in range(n):
        mark = "OK " if pred[i] == y[i] else "MISS"
        print(f"[{mark}] image {i:2d}: true = {to_letter(y[i])}  "
              f"predicted = {to_letter(pred[i])}")
    print(f"\n{n and correct}/{n} correct on this sample "
          f"({correct / n * 100:.1f}%)")

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.1))
    for ax, i in zip(np.array(axes).ravel(), range(n)):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ok = pred[i] == y[i]
        ax.set_title(f"{to_letter(y[i])} -> {to_letter(pred[i])}",
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    for ax in np.array(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Predictions on held-out test images — {name}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"Grid saved to {OUT}")


if __name__ == "__main__":
    main()
