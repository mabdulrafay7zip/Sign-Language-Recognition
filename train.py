"""Train and compare three classifiers on Sign Language MNIST.

Models: SVM (RBF), Random Forest, and an MLP neural network — all from
scikit-learn. The dataset ships a fixed train/test split (27,455 / 7,172),
which is used as-is so results are comparable with other published work.

Outputs (in results/ and models/):
  - accuracy_comparison.png, confusion_matrix.png, sample_predictions.png
  - results.txt  (accuracies + full classification report of the best model)
  - models/best_model.joblib
"""
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from common import LETTERS, LABEL_TO_LETTER, load_data, to_letter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def plot_accuracy_comparison(scores):
    names = list(scores.keys())
    vals = [scores[n] * 100 for n in names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, vals, color=["#3b6ea5", "#4c9f70", "#c0564f"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}%",
                 ha="center", fontsize=10, fontweight="bold")
    plt.ylabel("Test accuracy (%)")
    plt.title("Sign Language MNIST — model comparison (test set, n=7,172)")
    plt.ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "accuracy_comparison.png"), dpi=150)
    plt.close()


def plot_confusion(cm, title):
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(LETTERS)), LETTERS)
    plt.yticks(range(len(LETTERS)), LETTERS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_samples(X_test, y_test, y_pred, n=12):
    rng = np.random.default_rng(7)
    idx = rng.choice(len(X_test), size=n, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ok = y_pred[i] == y_test[i]
        ax.set_title(f"true {to_letter(y_test[i])} / pred {to_letter(y_pred[i])}",
                     color="green" if ok else "red", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample test predictions (best model)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sample_predictions.png"), dpi=150)
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"train {X_train.shape}, test {X_test.shape}, "
          f"classes {len(np.unique(y_train))}")

    models = {
        "SVM (RBF, C=10)": SVC(kernel="rbf", C=10, gamma="scale",
                               cache_size=1000, random_state=42),
        "Random Forest (300 trees)": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42),
        "MLP (256-128)": MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
            batch_size=256, max_iter=40, random_state=42),
    }

    scores, preds, fitted = {}, {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        scores[name] = acc
        preds[name] = pred
        fitted[name] = model
        print(f"{name}: test accuracy {acc * 100:.2f}%  "
              f"({time.time() - t0:.0f}s)")

    best_name = max(scores, key=scores.get)
    best_model = fitted[best_name]
    y_pred = preds[best_name]
    print(f"\nBest model: {best_name} — {scores[best_name] * 100:.2f}%")

    labels_sorted = sorted(LABEL_TO_LETTER)
    report = classification_report(
        y_test, y_pred, labels=labels_sorted,
        target_names=[LABEL_TO_LETTER[i] for i in labels_sorted],
        digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    plot_accuracy_comparison(scores)
    plot_confusion(cm, f"Confusion matrix — {best_name} "
                       f"({scores[best_name] * 100:.2f}% test accuracy)")
    plot_samples(X_test, y_test, y_pred)

    joblib.dump(best_model, os.path.join(MODELS, "best_model.joblib"))
    with open(os.path.join(MODELS, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    lines = [
        "Sign Language Recognition — results (Sign Language MNIST)",
        f"Train: {len(X_train)} images | Test: {len(X_test)} images | "
        f"Classes: 24 ASL letters (A-Y, excluding J and Z)",
        "",
        "Test accuracy by model:",
    ]
    for name, acc in scores.items():
        lines.append(f"  {name}: {acc * 100:.2f}%")
    lines += ["", f"Best model: {best_name}", "",
              "Classification report (best model, test set):", report]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "results.txt"), "w") as f:
        f.write(text)
    print("\n" + text)
    print("\nSaved: results/*.png, results/results.txt, models/best_model.joblib")


if __name__ == "__main__":
    main()
