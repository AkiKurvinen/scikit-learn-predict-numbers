"""
Load a previously saved model and print a single predicted digit.

Usage:
    python predict_from_saved.py 42    # predict excluded row 42 from semeion.data
"""

import sys
import pandas as pd
import joblib

DATA_PATH = "./dataset/semeion.data"        # update path if needed
MODEL_PATH = "digit_model.joblib"
SKIP_ROWS = 1000


def load_data(path, skip_rows=SKIP_ROWS):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[skip_rows:]
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
    if not 0 <= data_index < SKIP_ROWS:
        raise SystemExit(f"data_index must be between 0 and {SKIP_ROWS - 1}")

    model = joblib.load(MODEL_PATH)

    X, y = load_data(DATA_PATH, skip_rows=0)
    sample = X[data_index].reshape(1, -1)

    print_digit(sample)
    prediction = model.predict(sample)[0]
    print(f"I guess it's {prediction}")


if __name__ == "__main__":
    main()
