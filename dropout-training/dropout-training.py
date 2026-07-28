import numpy as np

def dropout(x, p = 0.5, rng = None):
    x = np.asarray(x, dtype = np.float64)
    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)
    dropout_pattern = np.where(rand >= p, 1.0 / (1 - p), 0.0)
    output = x * dropout_pattern 
    return output, dropout_pattern