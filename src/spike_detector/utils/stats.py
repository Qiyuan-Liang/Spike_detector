import numpy as np


def mean_std_count(arr):
    a = np.array(arr) if len(arr) > 0 else np.array([])
    if a.size > 0:
        return float(np.nanmean(a)), float(np.nanstd(a)), int(a.size)
    return np.nan, np.nan, 0
