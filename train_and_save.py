"""
Train a classifier on the Semeion handwritten digit dataset and save it to disk.

Usage:
    python train_and_save.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

DATA_PATH = "./dataset/semeion.data"   # update path if needed
MODEL_PATH = "digit_model.joblib"
# SKIP_ROWS = 500
SKIP_ROWS = 1580


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=SKIP_ROWS)
    X = df.iloc[:, :256].values
    y = df.iloc[:, -1].astype(int).values
    return X, y


def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
