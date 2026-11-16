from collections import Counter
import numpy as np

def mean_median_mode(x):
    x = np.asarray(x, dtype=float)
    counts = Counter(x.tolist())
    highest_frequency = max(counts.values())
    mode = min(value for value, count in counts.items() if count == highest_frequency)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode),
    }