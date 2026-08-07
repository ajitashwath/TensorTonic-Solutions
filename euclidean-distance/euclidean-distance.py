import numpy as np

def euclidean_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return float(np.sqrt(np.sum((x - y) ** 2)))