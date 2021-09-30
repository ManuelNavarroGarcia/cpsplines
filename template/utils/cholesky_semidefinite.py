import numpy as np
from scipy.linalg import ldl


def cholesky_semidef(A):
    LDL_decomp = ldl(A, lower=False)
    sqrt_decomp = np.sqrt(np.clip(np.diag(LDL_decomp[1]), a_min=0, a_max=1e16))
    return sqrt_decomp * LDL_decomp[0]