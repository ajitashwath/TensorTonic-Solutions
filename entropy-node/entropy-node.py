import numpy as np

def entropy_node(y):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts = True)
    p = counts.astype(np.float64) / y.size
    return float(-np.sum(p * np.log2(p)))