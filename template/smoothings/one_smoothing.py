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
    kron_tens_prod,
    penalization_term,
    matrix_by_transpose,
)

from template.utils.gcv import GCV, gcv_mat
from template.utils.simulator_optimize import Simulator
from template.utils.simulator_grid_search import print_grid_search_results

from template.utils.cholesky_semidefinite import cholesky_semidef


class NumericalError(Exception):
    pass


class OneSmoothing:
    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (40,),
        x_range: Optional[Dict[int, Tuple[Union[int, float]]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[Dict[str, Any]] = None,
        int_constraints: Optional[
            Dict[int, Dict[int, Dict[str, Union[int, float]]]]
        ] = None,
        pt_constraints: Optional[Dict[Tuple[int], Any]] = None,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.sp_method = sp_method
        self.sp_args = sp_args
        self.int_constraints = int_constraints
        self.pt_constraints = pt_constraints

    def _get_bspline_bases(self, x: Iterable[np.ndarray]) -> List[BsplineBasis]:

        bspline_bases = []
        if self.x_range is None:
            self.x_range = {}
        for i in range(len(self.deg)):
            x_min, x_max = np.min(x[i]), np.max(x[i])
            prediction_dict = {}
            if i in self.x_range.keys():
                pred_min, pred_max = min(self.x_range[i]), max(self.x_range[i])
                if pred_max > x_max:
                    prediction_dict["forward"] = pred_max
                if pred_min < x_min:
                    prediction_dict["backwards"] = pred_min
            bsp = BsplineBasis(
                deg=self.deg[i],
                xsample=x[i],
                n_int=self.n_int[i],
                prediction=prediction_dict,
            )
            bsp.get_matrix_B()
            bspline_bases.append(bsp)
        return bspline_bases

    def _fill_sp_args(self) -> dict:
        if self.sp_args is None:
            self.sp_args = {}
        self.sp_args["verbose"] = self.sp_args.get("verbose", False)
        if self.sp_method == "grid_search":
            self.sp_args["grid"] = self.sp_args.get(
                "grid", [[10 ** i for i in range(-2, 2)] for _ in range(len(self.deg))]
            )
            self.sp_args["parallel"] = self.sp_args.get("parallel", False)
            self.sp_args["n_jobs"] = self.sp_args.get("n_jobs", -2)
        elif self.sp_method == "optimizer":
            self.sp_args["sp_guess"] = self.sp_args.get(
                "sp_guess", [1 for _ in range(len(self.deg))]
            )
            self.sp_args["optim_method"] = self.sp_args.get("optim_method", "SLSQP")
            self.sp_args["optim_options"] = self.sp_args.get(
                "optim_options", {"ftol": 1e-12, "maxiter": 100}
            )
            self.sp_args["bounds"] = self.sp_args.get(
                "bounds", [[1e-10, 1e16] for _ in range(len(self.deg))]
            )
        else:
            raise ValueError(f"invalid `sp_method`: {self.sp_method}")

    def _get_obj_func_arrays(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        matrix_dict = {}
        matrix_dict["B"] = []
        matrix_dict["D_mul"] = []
        y_extended_dim = []
        for i, bsp in enumerate(self.bspline_bases):
            B = bsp.matrixB
            y_extended_dim.append(B.shape[0])
            matrix_dict["B"].append(B)
            matrix_dict["D_mul"].append(
                PenaltyMatrix(bspline=bsp).get_penalty_matrix(
                    **{"ord_d": self.ord_d[i]}
                )
            )

        y_extended = np.zeros(tuple(y_extended_dim))
        y_extended[get_idx_fitting_region(self.bspline_bases)] = y
        matrix_dict["y"] = y_extended
        return matrix_dict

    def _initialize_model(
        self, lin_term: np.ndarray, L_B: np.ndarray, L_D: Iterable[np.ndarray]
    ) -> mosek.fusion.Model:
        M = mosek.fusion.Model()
        # Build the objective function of the problem
        mos_obj_f = ObjectiveFunction(bspline=self.bspline_bases, model=M)
        sp = [M.parameter(f"sp_{i}", 1) for i in range(len(L_D))]
        mos_obj_f.create_obj_function(
            L_B=L_B,
            L_D=L_D,
            sp=sp,
            lin_term=lin_term,
        )

        if self.int_constraints is not None:
            S_dict = {}
            for i, bsp in enumerate(self.bspline_bases):
                S_dict[i] = bsp.get_matrices_S()
            for var_name in self.int_constraints.keys():
                for deriv in self.int_constraints[var_name].keys():
                    S_dict_copy = S_dict.copy()
                    cons = IntConstraints(
                        bspline=self.bspline_bases,
                        var_name=var_name,
                        derivative=deriv,
                        constraints=self.int_constraints[var_name][deriv],
                    )
                    cons.interval_cons(
                        var_dict=mos_obj_f.var_dict, model=M, S_dict=S_dict_copy
                    )
        else:
            self.int_constraints = {}

        if self.pt_constraints is not None:
            for deriv, info in self.pt_constraints.items():
                cons2 = PointConstraints(
                    pts=info[0],
                    value=info[1],
                    derivative=deriv,
                    bspline=self.bspline_bases,
                    tolerance=info[2],
                )
                cons2.point_cons(var_dict=mos_obj_f.var_dict, model=M)
        else:
            self.pt_constraints = {}

        return M

    def _get_sp_grid_search(
        self,
        B_weighted: Iterable[np.ndarray],
        gcv_tuple: Iterable[np.ndarray],
        y: np.ndarray,
    ) -> Tuple[Union[int, float]]:
        # Computes all the possible combinations for the smoothing parameters
        iter_sp = list(itertools.product(*self.sp_args["grid"]))
        if self.sp_args["parallel"] == True:
            gcv_list = Parallel(n_jobs=self.sp_args["n_jobs"])(
                delayed(GCV)(sp, B_weighted, gcv_tuple, y) for sp in iter_sp
            )
        else:
            gcv_list = []
            for sp in iter_sp:
                gcv = GCV(
                    sp=sp,
                    B_weighted=B_weighted,
                    qua_term=gcv_tuple,
                    y=y,
                )
                gcv_list.append(gcv)
        if self.sp_args["verbose"] == True:
            print_grid_search_results(input_values=iter_sp, objective=gcv_list)

        return iter_sp[gcv_list.index(min(gcv_list))]

    def _get_sp_optimizer(
        self,
        B_weighted: Iterable[np.ndarray],
        gcv_tuple: Iterable[np.ndarray],
        y: np.ndarray,
    ) -> Tuple[Union[int, float]]:
        gcv_sim = Simulator(GCV)
        best_sp = scipy.optimize.minimize(
            gcv_sim.simulate if self.sp_args["verbose"] else GCV,
            self.sp_args["sp_guess"],
            args=(
                B_weighted,
                gcv_tuple,
                y,
            ),
            callback=gcv_sim.callback if self.sp_args["verbose"] else None,
            method=self.sp_args["optim_method"],
            bounds=self.sp_args["bounds"],
            options=self.sp_args["optim_options"],
        ).x
        return best_sp

    def fit(self, x: Iterable[np.ndarray], y: np.ndarray):

        if len({len(i) for i in [self.deg, self.ord_d, self.n_int, x]}) != 1:
            raise ValueError(
                "The lengths of `deg`, `ord_d`, `n_int` and `x` must agree."
            )

        if self.sp_method not in ["grid_search", "optimizer"]:
            raise ValueError(f"invalid `sp_method`: {self.sp_method}")

        self.bspline_bases = self._get_bspline_bases(x=x)

        _ = self._fill_sp_args()

        matrix_dict = self._get_obj_func_arrays(y=y)

        # Auxiliary matrices
        B_weighted = get_weighted_B(bspline_bases=self.bspline_bases)
        matrix_dict["B_mul"] = list(map(matrix_by_transpose, B_weighted))
        lin_term = np.multiply(
            -2, kron_tens_prod([mat.T for mat in B_weighted], matrix_dict["y"])
        ).flatten()

        # Compute the Cholesky factorization (A = L @ L.T)
        L_B = reduce(
            fast_kronecker_product, list(map(cholesky_semidef, matrix_dict["B_mul"]))
        )

        L_D = penalization_term(
            matrices=list(map(cholesky_semidef, matrix_dict["D_mul"]))
        )

        # Initialize the model
        M = self._initialize_model(lin_term=lin_term, L_B=L_B, L_D=L_D)
        model_params = {"theta": M.getVariable("theta")}
        for i in range(len(self.bspline_bases)):
            model_params[f"sp_{i}"] = M.getParameter(f"sp_{i}")

        gcv_tuple = gcv_mat(B_mul=matrix_dict["B_mul"], D_mul=matrix_dict["D_mul"])

        if self.sp_method == "grid_search":
            self.best_sp = self._get_sp_grid_search(
                B_weighted=B_weighted, gcv_tuple=gcv_tuple, y=matrix_dict["y"]
            )
        else:
            self.best_sp = self._get_sp_optimizer(
                B_weighted=B_weighted, gcv_tuple=gcv_tuple, y=matrix_dict["y"]
            )

        theta_shape = model_params["theta"].getShape()
        for i, sp in enumerate(self.best_sp):
            model_params[f"sp_{i}"].setValue(sp)
        try:
            M.solve()
            self.sol = model_params["theta"].level().reshape(theta_shape)
            # Compute the fitted values of the response variable
            self.y_fitted = kron_tens_prod([mat for mat in matrix_dict["B"]], self.sol)
        except mosek.fusion.SolutionError as e:
            raise NumericalError(
                f"The solution for the smoothing parameter {self.best_sp} "
                f"could not be found due to numerical issues. The original error "
                f"was: {e}"
            )

        return None

    def predict(self, x: Iterable[np.ndarray]) -> np.ndarray:
        if len([v for v in x if len(v) > 0]) < len(x):
            return np.array([])
        else:
            x_min = np.array([np.min(v) for v in x])
            x_max = np.array([np.max(v) for v in x])
            bsp_min = np.array([bsp.knots[bsp.deg] for bsp in self.bspline_bases])
            bsp_max = np.array([bsp.knots[-bsp.deg] for bsp in self.bspline_bases])
            if (x_min < bsp_min).sum() > 0:
                raise ValueError(f"Algo izquierda")
            if (x_max > bsp_max).sum() > 0:
                raise ValueError("Algo derecha")
            B_weighted = []
            for i, bsp in enumerate(self.bspline_bases):
                B_weighted.append(bsp.bspline_basis.derivative(nu=0)(x[i]))
            return kron_tens_prod([mat for mat in B_weighted], self.sol)
