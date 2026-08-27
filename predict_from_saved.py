"""
Load a previously saved model and print a single predicted digit.

Usage:
    python predict_from_saved.py 42    # predict row 42 from semeion.data
"""

import sys
import pandas as pd
import joblib

DATA_PATH = "./dataset/semeion.data"        # update path if needed
MODEL_PATH = "digit_model.joblib"


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    X = df.iloc[:, :256].values
    y = df.iloc[:, -1].astype(int).values
    return X, y


def print_digit(sample):
    for row in sample.reshape(16, 16):
        print("".join("■" if pixel > 0.5 else "□" for pixel in row))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_from_saved.py <data_index>")

    data_index = int(sys.argv[1])

    model = joblib.load(MODEL_PATH)

    X, y = load_data(DATA_PATH)
    if not 0 <= data_index < len(X):
        raise SystemExit(f"data_index must be between 0 and {len(X) - 1}")

    sample = X[data_index].reshape(1, -1)

    print_digit(sample)
    prediction = model.predict(sample)[0]
    print(f"I guess it's {prediction}")


if __name__ == "__main__":
    main()
