import itertools
import numpy as np
import mosek.fusion
import scipy
from joblib import delayed, Parallel
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from functools import reduce

from template.mosek_functions.obj_function import ObjectiveFunction
from template.mosek_functions.interval_constraints import IntConstraints
from template.mosek_functions.point_constraints import PointConstraints
from template.psplines.bspline_basis import BsplineBasis
from template.psplines.penalty_matrix import PenaltyMatrix
from template.utils.weighted_b import (
    get_weighted_B,
    get_idx_fitting_region,
    get_weighted_B,
)
from template.utils.fast_kron import (
    fast_kronecker_product,
    matrix_by_tensor_product,
    penalization_term,
    matrix_by_transpose,
)

from template.utils.gcv import GCV, gcv_mat
from template.utils.simulator_optimize import Simulator
from template.utils.simulator_grid_search import print_grid_search_results

from template.utils.cholesky_semidefinite import cholesky_semidef


class NumericalError(Exception):
    pass


class Adaptive1D:
    def __init__(
        self,
        bdeg: int = 3,
        ord_d: int = 2,
        b_int: int = 40,
        spdeg: int = 3,
        sp_int: int = 10,
        int_constraints: Optional[
            Dict[int, Dict[int, Dict[str, Union[int, float]]]]
        ] = None,
        max_iter: int = 200,
        tol: Union[int, float] = 1e-3,
        phi_guess: Optional[Union[int, float]] = None,
        tau_guess: Optional[Iterable[Union[int, float]]] = None,
    ):
        self.bdeg = bdeg
        self.ord_d = ord_d
        self.b_int = b_int
        self.spdeg = spdeg
        self.sp_int = sp_int
        self.int_constraints = int_constraints
        self.max_iter = max_iter
        self.tol = tol
        self.phi_guess = phi_guess
        self.tau_guess = tau_guess

    def _get_obj_func_arrays(
        self, x: np.ndarray
    ) -> Dict[str, Union[BsplineBasis, np.ndarray]]:

        bbasis = BsplineBasis(deg=self.bdeg, n_int=self.b_int, xsample=x)
        bbasis.get_matrix_B()
        D = np.diff(
            np.eye(bbasis.matrixB.shape[1] + self.ord_d, dtype=np.int32), n=self.ord_d
        )[self.ord_d : -self.ord_d, :]

        U_Z = np.linalg.solve(D @ D.T, D).T
        Z = bbasis.matrixB @ U_Z
        u, _, _ = np.linalg.svd(D.T @ D)
        U_X = u[:, -self.ord_d :]
        X = bbasis.matrixB @ U_X

        spbasis = BsplineBasis(
            deg=self.spdeg,
            n_int=self.sp_int,
            xsample=np.linspace(1, Z.shape[1], Z.shape[1]) / Z.shape[1],
        )
        spbasis.get_matrix_B()

        return {
            "bbasis": bbasis,
            "spbasis": spbasis,
            "D": D,
            "U_Z": U_Z,
            "U_X": U_X,
            "Z": Z,
            "X": X,
        }

    def _deviance(self, C, G, n, phi, ssr, edf):
        _, log_det_C = np.linalg.slogdet(C)
        _, log_det_G = np.linalg.slogdet(G)
        return log_det_C + log_det_G + n * np.log(phi) + ssr / phi + edf

    def _find_best_sp(self, y: np.ndarray):
        phi = 1 if self.phi_guess is None else self.phi_guess
        tau = (
            np.ones(self.obj_utils["spbasis"].matrixB.shape[1])
            if self.tau_guess is None
            else self.tau_guess
        )

        S = np.concatenate((self.obj_utils["X"], self.obj_utils["Z"]), axis=1)
        C = S.T @ S
        u = np.dot(y, S)

        devold = np.inf

        for _ in range(self.max_iter):
            print(devold)
            Ginv = np.einsum("ij,j->i", self.obj_utils["spbasis"].matrixB, 1 / tau)

            H = np.multiply(1 / phi, C) + np.diag(
                np.concatenate([np.zeros(self.ord_d), Ginv])
            )
            try:
                Hinv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                Hinv = np.linalg.pinv(H)

            b = np.multiply(1 / phi, np.dot(Hinv, u))

            G = 1 / Ginv
            aux = G - np.diag(Hinv[self.ord_d :, self.ord_d :])
            b_random_square = np.square(b[self.ord_d :])

            ed = np.clip(
                a=np.divide(np.dot(aux, self.obj_utils["spbasis"].matrixB), tau),
                a_min=1e-10,
                a_max=None,
            )
            tau = np.clip(
                a=np.divide(
                    np.dot(b_random_square, self.obj_utils["spbasis"].matrixB), ed
                ),
                a_min=1e-10,
                a_max=None,
            )

            err = y - np.dot(S, b)
            ssr = np.linalg.norm(err) ** 2
            phi = ssr / (len(y) - ed.sum() - self.ord_d)

            dev = self._deviance(
                C=H,
                G=np.diag(G),
                n=len(y),
                phi=phi,
                ssr=ssr,
                edf=np.dot(b_random_square, Ginv),
            )

            if np.abs(devold - dev) < self.tol:
                break
            devold = dev.copy()
        self.deviance = dev
        self.best_sp = phi / tau
        return {"y": S @ b, "sol": b}

    def _initialize_model(
        self, lin_term: np.ndarray, L_B: np.ndarray, L_D: Iterable[np.ndarray]
    ) -> mosek.fusion.Model:

        M = mosek.fusion.Model()
        # Create the variables of the optimization problem
        mos_obj_f = ObjectiveFunction(bspline=(self.obj_utils["bbasis"],), model=M)
        # For each axis, a smoothing parameter is needed
        sp = [M.parameter(f"sp_{i}", 1) for i in range(len(L_D))]
        # Build the objective function of the problem
        mos_obj_f.create_obj_function(
            L_B=L_B,
            L_D=L_D,
            sp=sp,
            lin_term=lin_term,
        )
        matrices_S = {}
        # If the are interval constraints, construct the matrices S
        for i, bsp in enumerate((self.obj_utils["bbasis"],)):
            matrices_S[i] = bsp.get_matrices_S()
        # Iterate for every variable with constraints and for every
        # derivative order
        for var_name in self.int_constraints.keys():
            for deriv in self.int_constraints[var_name].keys():
                matrices_S_copy = matrices_S.copy()
                # Build the interval constraints
                cons = IntConstraints(
                    bspline=(self.obj_utils["bbasis"],),
                    var_name=var_name,
                    derivative=deriv,
                    constraints=self.int_constraints[var_name][deriv],
                )
                cons.interval_cons(
                    var_dict=mos_obj_f.var_dict, model=M, matrices_S=matrices_S_copy
                )

        return M

    def fit(self, x: np.ndarray, y: np.ndarray):

        self.obj_utils = self._get_obj_func_arrays(x=x)

        unconstrained_results = self._find_best_sp(y=y)
        if self.int_constraints is None:
            self.sol = unconstrained_results["sol"]
            self.y_fitted = unconstrained_results["y"]
        # else:
        #     B_mul = (
        #         self.obj_utils["bbasis"].matrixB.T @ self.obj_utils["bbasis"].matrixB
        #     )
        #     L_B = cholesky_semidef(B_mul)
        #     L_D = tuple(
        #         [
        #             cholesky_semidef(
        #                 self.obj_utils["D"].T
        #                 @ np.diag(self.obj_utils["spbasis"].matrixB[:, i])
        #                 @ self.obj_utils["D"]
        #             )
        #             for i in range(self.obj_utils["spbasis"].matrixB.shape[1])
        #         ]
        #     )
        #     lin_term = np.multiply(-2, np.dot(y, self.obj_utils["bbasis"].matrixB))

        #     M = self._initialize_model(lin_term=lin_term, L_B=L_B, L_D=L_D)
        #     model_params = {"theta": M.getVariable("theta")}
        #     # Set the smoothing parameters vector as the optimal obtained in the
        #     # unconstrained setting
        #     for i, sp in enumerate(self.best_sp):
        #         model_params[f"sp_{i}"] = M.getParameter(f"sp_{i}")
        #         model_params[f"sp_{i}"].setValue(sp)
        #     try:
        #         # Solve the problem
        #         M.solve()
        #         # Extract the fitted decision variables of the B-spline expansion
        #         self.sol = model_params["theta"].level()
        #         # Compute the fitted values of the response variable
        #         self.y_fitted = np.dot(
        #             self.obj_utils["bbasis"].matrixB,
        #             self.sol,
        #         )
        #     except mosek.fusion.SolutionError as e:
        #         raise NumericalError(
        #             f"The solution for the smoothing parameter {self.best_sp} "
        #             f"could not be found due to numerical issues. The original error "
        #             f"was: {e}"
        #         )
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        Bp = self.obj_utils["bbasis"].bspline_basis.derivative(nu=0)(x)
        design_m = Bp @ np.concatenate(
            (self.obj_utils["U_X"], self.obj_utils["U_Z"]), axis=1
        )
        yp = np.dot(design_m, self.sol)
        return yp
