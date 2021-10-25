import operator
from itertools import tee

import numpy as np


def is_sorted(iterable, compare=operator.le):
    a, b = tee(iterable)
    next(b, None)
    if all(map(compare, a, b)):
        return None
    else:
        return np.argsort(iterable)


print(is_sorted(np.array([1, 2, 4, 3, 5, 8])))
