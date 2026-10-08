"""Shared helpers: label mapping and dataset loading for Sign Language MNIST."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")

# Sign MNIST labels run 0-24 with 9 skipped: label 9 would be 'J' and 25 would
# be 'Z'; both are motion signs in ASL, so the dataset holds 24 static letters.
LABEL_TO_LETTER = {i: chr(ord("A") + i) for i in range(25) if i != 9}
LETTERS = [LABEL_TO_LETTER[i] for i in sorted(LABEL_TO_LETTER)]


def load_data():
    """Return X_train, y_train, X_test, y_test with pixels scaled to [0, 1]."""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    X_train = train.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    X_test = test.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    return X_train, y_train, X_test, y_test


def to_letter(label: int) -> str:
    return LABEL_TO_LETTER[int(label)]
