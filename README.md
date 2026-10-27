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

## Interpreting the webcam demo output

- The window shows `Letter: X (NN%)` at the top: `X` is the predicted letter for the current frame and `NN%` is the model's confidence for that frame (shown when the model supports `predict_proba`).
- That confidence is **not** the test accuracy: 83.71% was measured once over all 7,172 test images, while the webcam percentage changes frame by frame as your hand, lighting, or position in the box changes.
- If the letter flickers or confidence stays low, re-centre your hand inside the on-screen box and hold the sign still for a moment before reading too much into a single frame.
- Press `Q` or `Esc` to quit the demo.

## Interpreting the predict_test.py output

- The script prints one line per image: `[OK ]` when the prediction matches the true letter and `[MISS]` when it does not, in the form `true = X  predicted = Y`.
- The final line, `X/N correct on this sample (NN.N%)`, covers only the first N test images (16 by default — pass a number, e.g. `python predict_test.py 32`, to change it). A small sample like this will usually differ from the model's 83.71% accuracy over the full 7,172-image test set, so do not read it as a new overall score.
- It also saves a labelled grid to `results/prediction_demo.png`: each tile is titled `true -> predicted`, in green when correct and red when wrong, which makes the model's confusions easy to spot.
- The script needs the trained model first: if `models/best_model.joblib` is missing it exits with `Model not found — run train.py first.`

## Interpreting the train.py output

- Run `python train.py` only after `python download_data.py`: train.py loads the data first, so without the CSVs in `data/` it cannot start training.
- The first line reports the loaded data, in the form `train (27455, 784), test (7172, 784), classes 24`.
- It then prints one line per model, in the form `Name: test accuracy NN.NN%  (Ns)`, for the SVM, the Random Forest, and the MLP. The percentage is accuracy on the held-out test set, and the number in parentheses is how many seconds that model took.
- Next comes `Best model: Name — NN.NN%`. The best model is whichever of the three scored highest on the test set in that run.
- The script then prints the full results text and saves it to `results/results.txt`, including the accuracy of each model and the per-class classification report for the best model only.
- It also saves `results/accuracy_comparison.png`, `results/confusion_matrix.png`, `results/sample_predictions.png`, `models/best_model.joblib`, and `models/best_model_name.txt`.

## Interpreting the download_data.py output

- Run `python download_data.py` before `python train.py`: it fetches the two dataset CSVs into `data/` — `sign_mnist_train.csv` (27,455 rows) and `sign_mnist_test.csv` (7,172 rows) — which are not included in this repository.
- For each file it prints `downloading <url> ...` and then `saved <path> (NN.N MB)` once that file is written; wait for both "saved" lines before running train.py.
- On a repeat run, a file that is already in `data/` prints `already present: <path>` and is skipped, not re-downloaded — so re-running the script is safe and does not waste a second download.
- The files come from the public GitHub mirror named in the script; the original dataset is Kaggle's Sign Language MNIST, and each row is `label, pixel1 ... pixel784` (a 28x28 grayscale image, values 0-255).

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
