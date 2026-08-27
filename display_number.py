import sys
import pandas as pd

DATA_PATH = "./dataset/semeion.data"


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    X = df.iloc[:, :256].values
    return X


def print_digit(sample):
    for row in sample.reshape(16, 16):
        print("".join("■" if pixel > 0.5 else "□" for pixel in row))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python display_number.py <data_index>")

    data_index = int(sys.argv[1])

    X = load_data(DATA_PATH)
    print_digit(X[data_index])


if __name__ == "__main__":
    main()
