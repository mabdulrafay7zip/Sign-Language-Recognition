"""Download the Sign Language MNIST dataset (no account / auth needed).

Source: public GitHub mirror of the Kaggle 'Sign Language MNIST' dataset
(originally by datamunge on Kaggle). Files:
  - sign_mnist_train.csv  (27,455 rows)
  - sign_mnist_test.csv   (7,172 rows)
Each row is: label, pixel1 ... pixel784  (28x28 grayscale, values 0-255).
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/gchilingaryan/Sign-Language/master"
FILES = ["sign_mnist_train.csv", "sign_mnist_test.csv"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest):
            print(f"already present: {dest}")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
