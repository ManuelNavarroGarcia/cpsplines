import numpy as np


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
