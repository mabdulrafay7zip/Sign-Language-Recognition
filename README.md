# Sign Language Recognition — ASL Alphabet Classifier

A machine learning project that recognises **American Sign Language (ASL) alphabet hand signs** from images. Given a 28×28 grayscale image of a hand sign, the model predicts which letter it represents — **24 static letters, A–Y** (J and Z are excluded because in ASL they are motion signs, not static poses).

This is the natural next step from my earlier Handwritten Character Recognition project: the same image-classification pipeline, applied to a real-world accessibility problem — helping bridge communication with the deaf and hard-of-hearing community.

## Dataset

**Sign Language MNIST** (originally published on Kaggle by *datamunge*), a public GitHub mirror is used so no account is needed to download it.

| | Images | Format |
|---|---|---|
| Training set | 27,455 | 28×28 grayscale, flattened to 784 pixels (0–255) |
| Test set | 7,172 | same |
| Classes | 24 | ASL letters A–Y, excluding J (label 9) and Z (label 25) |

Each CSV row is `label, pixel1 … pixel784`. The dataset ships a fixed train/test split, which is used as-is so results are comparable with other published work.

Download it with:

```bash
python download_data.py
```

## Approach

1. **Preprocessing** — pixel values scaled from 0–255 to [0, 1]; no other transformation (the images are already centred and cropped).
2. **Three models trained and compared** (all scikit-learn):
   - **SVM** with an RBF kernel (C = 10)
   - **Random Forest** (300 trees)
   - **MLP neural network** (hidden layers 256 → 128, Adam)
3. **Evaluation** on the held-out official test set: accuracy, per-class classification report, and a confusion matrix.
4. **Best model saved** with joblib and used by both demo scripts.

## Results (actual run, official test set of 7,172 images)

| Model | Test accuracy |
|---|---|
| **SVM (RBF, C=10)** | **83.71%** ✅ best |
| Random Forest (300 trees) | 82.36% |
| MLP (256–128) | 76.80% |

The SVM also reached perfect or near-perfect recall on several letters (A: 100%, P: 100%, B: 99.1%, E: 99.8%), while visually similar signs such as R, U, V and K were the hardest — see the confusion matrix in `results/`.

> Note: the MLP stopped at its 40-iteration cap before fully converging, which partly explains its lower score. These are honest single-run numbers on a CPU-only machine — a CNN would be expected to score higher, and is listed under Future Work.

Plots in `results/`:
- `accuracy_comparison.png` — the table above as a chart
- `confusion_matrix.png` — 24×24 confusion matrix of the best model
- `sample_predictions.png` / `prediction_demo.png` — example test images with true vs predicted letters
- `results.txt` — full numbers and the per-class classification report

## Project structure

```
sign-language-project/
├── common.py            # label mapping (0-24 -> letters) + data loading
├── download_data.py     # fetches the dataset CSVs (no account needed)
├── train.py             # trains SVM, Random Forest, MLP; saves plots + best model
├── predict_test.py      # runs the saved model on held-out test images
├── webcam_demo.py       # live webcam demo (OpenCV)
├── requirements.txt
├── data/                # dataset CSVs (gitignored, ~105 MB — use download_data.py)
├── models/              # best_model.joblib (SVM, ~44 MB) + best_model_name.txt
└── results/             # plots, results.txt, training log
```

## How to run

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the data
python download_data.py

# 3. Train all three models and generate the results
python train.py

# 4a. Test the saved model on held-out images (no camera needed)
python predict_test.py           # add a number, e.g. 16, for more images

# 4b. Live webcam demo — hold your hand in the on-screen box signing a letter
python webcam_demo.py            # press Q to quit
```

## Webcam demo — honest limitations

The model was trained on clean, centred Sign MNIST images with dark backgrounds. A raw webcam feed looks quite different (lighting, skin tones, busy backgrounds), so live predictions will be noticeably less accurate than the 83.71% test score. A plain dark background and good lighting help. Improving this (hand detection/cropping, background removal, training on real photos) is future work, not a solved part of this project.

## Tech stack

Python 3.12 · scikit-learn (SVM, Random Forest, MLP) · NumPy · pandas · matplotlib · joblib · OpenCV (webcam demo)

## Future work

- CNN model for higher accuracy (the dataset is image data — a convolutional network is the natural fit)
- Train on real hand photos (e.g. the ASL Alphabet dataset) to close the webcam domain gap
- Hand-landmark features (e.g. MediaPipe) instead of raw pixels for the live demo
- Word-level recognition by stringing letter predictions together

## Author

**Muhammad Abdul Rafay** — BS Artificial Intelligence, Air University, Islamabad
GitHub: [github.com/mabdulrafay7zip](https://github.com/mabdulrafay7zip)
