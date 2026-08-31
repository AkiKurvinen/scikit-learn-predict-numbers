"""
Basic data-analytics example: cluster the Semeion handwritten digit dataset with KMeans.

Usage:
    python classification.py
"""

import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = "./dataset/semeion.data"
CLUSTERS_MIN = 10
CLUSTERS_MAX = 10

def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    X = df.iloc[:, :256].values
    return X


def find_best_k(X, k_min, k_max):
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k={k}: silhouette score={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def main():
    X = load_data(DATA_PATH)

    n_clusters = find_best_k(X, CLUSTERS_MIN, CLUSTERS_MAX)
    print(f"\nBest number of clusters: {n_clusters}")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = model.fit_predict(X)

    print(f"Clustered {len(X)} digits into {n_clusters} clusters")

    print(f"\nSample count per cluster (0-{n_clusters - 1}):")
    print(pd.Series(clusters).value_counts().sort_index())

    if len(sys.argv) == 2:
        index = int(sys.argv[1])
        print(f"\nSample {index} was assigned to cluster {clusters[index]}")


if __name__ == "__main__":
    main()
