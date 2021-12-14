import itertools
from typing import Dict, Iterable, List, Tuple, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from scipy.special import comb, factorial


class PDFConstraints:
    def __init__(
        self,
        bspline: Iterable[BsplineBasis],
        var_name: int,
    ):
        self.bspline = bspline
        self.var_name = var_name

    def pdf_cons(
        self,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
    ):
        knots_dict = {}
        for j, bsp in enumerate(self.bspline):
            knots_dict[j] = bsp.knots[bsp.deg : -bsp.deg]
        knots_iter = list(itertools.product(*knots_dict.values()))
        return None
