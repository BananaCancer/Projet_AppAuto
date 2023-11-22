# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:48:13 2023

@author: basil
"""
import numpy as np

algorithms = {
    "Kmeans": {
        "k": np.arange(3, 6)
    },
    "DBSCAN": {
        "eps": [1, 2, 3],
        "min_samples": np.arange(2, 9)
    },
    "Hclust": {
        "k": np.arange(3, 6),
        "metrics": ["euclidean", "l1", "l2", "manhattan", "cosine"],
        "linkage": ["complete", "average", "single"],
        "ward": True
    },
    "GMM": {
        "k": np.arange(3, 6),
        "cov_type": ['tied', 'spherical'] # unused: full, diag
    },
    "Spectral": {
        "k": np.arange(3, 6),
        "eigen_solvers": ["arpack", "lobpcg", "amg"],
        "affinities": ["rbf"], # unused: nearest_neighbors
        "n_neighbors": [2, 5, 10, 25, 50]
    }
}

def getAlgorithms(list_algorithms):
    return {key: algorithms[key] for key in list_algorithms}
    