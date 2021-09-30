import numpy as np


def get_idx_fitting_region(list_bs_basis: list):
    return tuple(
        slice(bsp.int_back, bsp.int_back + len(bsp.xsample), None)
        for bsp in list_bs_basis
    )


def fast_B_weighted(list_bs_basis: list):
    weighted_mat = []
    idx_fitting_region = get_idx_fitting_region(list_bs_basis=list_bs_basis)
    for i in range(len(list_bs_basis)):
        B_weighted = np.zeros((list_bs_basis[i].matrixB.shape))
        slice_i = idx_fitting_region[i]
        B_weighted[slice_i, :] = list_bs_basis[i].matrixB[slice_i, :]
        weighted_mat.append(B_weighted)
    return weighted_mat
