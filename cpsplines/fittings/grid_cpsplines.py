import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mosek.fusion
import numpy as np
import pandas as pd
import scipy
from cpsplines.mosek_functions.interval_constraints import IntConstraints
from cpsplines.mosek_functions.obj_function import ObjectiveFunction
from cpsplines.mosek_functions.pdf_constraints import PDFConstraint
from cpsplines.mosek_functions.point_constraints import PointConstraints
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.fast_kron import matrix_by_tensor_product, matrix_by_transpose
from cpsplines.utils.gcv import GCV
from cpsplines.utils.normalize_data import DataNormalizer
from cpsplines.utils.rearrange_data import scatter_to_grid
from cpsplines.utils.simulator_grid_search import print_grid_search_results
from cpsplines.utils.simulator_optimize import Simulator
from cpsplines.utils.timer import timer
from cpsplines.utils.weighted_b import get_idx_fitting_region, get_weighted_B
from joblib import Parallel, delayed
from statsmodels.genmod.families.family import Binomial, Family, Gaussian, Poisson


class NumericalError(Exception):
    pass


class GridCPsplines:

    """
    Create the constrained P-splines model when data are in array form. The
    objective function of the optimization problem is defined using the approach
    of Currie et al. (2006) and the constraints are imposed over the whole
    domain with univariate data and over a finite set of curves (coinciding with
    the position of the knot sequences) otherwise.

    Parameters
    ----------
    deg : Iterable[int], optional
        The polynomial degree vector of the B-spline bases. All the elements
        must be non-negative integers. By default, (3,).
    ord_d : Iterable[int], optional
        The difference order vector of the penalty matrices. All the elements
        must be non-negative integers. By default, (2,).
    n_int : int, optional
        The vector of number of equal intervals which the fitting region along
        each dimension is split. All the elements must be non-negative integers.
        By default, (40,).
    x_range : Optional[Dict[int, Tuple[Union[int, float]]]], optional
        A dictionary containing the most extreme values that the extended
        B-spline basis needs to capture along each dimension. The keys are index
        of the regressor and the values are tuples containing the most extreme
        values. These values will only be considered when they are located
        outside the fitting region. One or two may be passed by variable. If
        None, it is intended that no extra range is needed in any of the axes.
        By default, None.
    sp_method : str, optional
        The method used to estimate the smoothing parameter as the minimizer of
        the Generalized Cross Validation criterium. It may be either `optimizer`
        (to use minimization algorithms in scipy.optimize.minimize) or
        `grid_search` (to select the smoothing parameters over a grid of
        candidates). By default, "optimizer".
    sp_args : Optional[Dict[str, Any]], optional
        The arguments employed on the chosen `sp_method`.
        1. If `grid_search` is selected, they are:
        - "grid" (Iterable[Iterable[Union[int, float]]]) : The smoothing
        parameter candidates along each axis. By default, (0.01, 0.1, 1, 10) on
        each direction.
        - "parallel" (bool) : If True, the different combinations of smoothing
        parameters are run in parallel using `joblib` module. By default, False.
        - "n_jobs" (int) : The number of jobs to run in parallel. By default,
        -2.
        - "top_n" (Optional[int]): If not None, prints the best `n` combinations
        of smoothing parameters.
        2. If `optimizer` is selected, they are:
        - "verbose" (bool) : If True, a callback printing the solution and the
        values at each iteration is used. By default, False.
        - The optional arguments and the first guess from
        scipy.optimize.minimize can be passed on this dictionary. By default,
        the SLSQP solver with a vector of ones as first guess is used. Also, the
        smoothing parameters are constrained to be in (1e-10, 1e16) since they
        must be non-negative.
    family : str, optional
        The specific exponential family distribution where the response variable
        belongs to. By default, "gaussian" (normal distribution).
    int_constraints : Dict[int, Dict[int, Dict[str, Union[int, float]]]]],
    optional
        A nested dictionary containing the interval constraints to be enforced.
        The keys of the dictionary are the indexes of the variables. The values
        are dictionaries, whose keys are the order of the derivative. The values
        are dictionareis, whose keys are the signs of the constraints (either
        "+" or "-") and the values are numbers denoting the upper or lower
        threshold of the constraints. By default, None.
    pt_constraints : Optional[Dict[Tuple[int], Any]], optional
        A dictionary containing the point constraints to be enforced. The keys
        of the dictionary are tuples representing the order of the derivatives
        where the constraints acts on. The values are tuples that must contain
        the items (in this order):
        - An array of unidimensional arrays with the coordinates of the points
        where the value needs to be fixed.
        - An array with the values of the derivative to be enforced.
        - A number corresponding to the tolerancea allowed in the constraint.
    pdf_constraint : bool, optional
        A boolean indicating whether the fitted hypersurface must satisfy
        Probability Density Function (PDF) conditions, i.e., it is non-negative
        and it integrates to one.

    Attributes
    ----------
    bspline_bases : List[BsplineBasis]
        The list of B-spline bases on each axis resulting from
        `_get_bspline_bases`.
    sol : np.ndarray
        The fitted decision variables of the B-spline expansion.
    y_fitted : np.ndarray
        The fitted values for the response variable.

    References
    ----------
    - Currie, I. D., Durban, M., & Eilers, P. H. (2006). Generalized linear
      array models with applications to multidimensional smoothing. Journal of
      the Royal Statistical Society: Series B (Statistical Methodology), 68(2),
      259-280.
    """

    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (40,),
        x_range: Optional[Dict[int, Tuple[Union[int, float]]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[Dict[str, Any]] = None,
        family: str = "gaussian",
        int_constraints: Optional[
            Dict[int, Dict[int, Dict[str, Union[int, float]]]]
        ] = None,
        pt_constraints: Optional[Dict[Tuple[int], Any]] = None,
        pdf_constraint: bool = False,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.sp_method = sp_method
        self.sp_args = sp_args
        self.family = self._get_family(family)
        self.int_constraints = int_constraints
        self.pt_constraints = pt_constraints
        self.pdf_constraint = pdf_constraint

    @staticmethod
    def _get_family(family: str) -> Family:
        """Given a distribution name from a distribution belonging to the
        exponential family, gets the corresponding Family object from
        statsmodels.

        Parameters
        ----------
        family : str
            The name of the distribution. It must be either "gaussian",
            "poisson" or "binomial"

        Returns
        -------
        Family
            The Family object from statsmodels corresponding to the
            distribution.

        Raises
        ------
        ValueError
            Raise an error when the input `family` is not implemented.
        """
        if family == "gaussian":
            family_statsmodels = Gaussian()
        elif family == "poisson":
            family_statsmodels = Poisson()
        elif family == "binomial":
            family_statsmodels = Binomial()
        else:
            raise ValueError(f"Family {family} is not implemented.")
        family_statsmodels.name = family
        return family_statsmodels

    def _get_bspline_bases(self, x: Iterable[np.ndarray]) -> List[BsplineBasis]:

        """
        Construct the B-spline bases on each axis.

        Parameters
        ----------
        x : Iterable[np.ndarray]
            The covariates samples.

        Returns
        -------
        List[BsplineBasis]
            The list of B-spline bases on each axis.
        """

        bspline_bases = []
        if self.x_range is None:
            self.x_range = {}
        for i in range(len(self.deg)):
            # Get the maximum and minimum of the fitting regions
            x_min, x_max = np.min(x[i]), np.max(x[i])
            prediction_dict = {}
            if i in self.x_range.keys():
                # If the values in `x_range` are outside the fitting region,
                # include them in the `prediction` argument of the BsplineBasis
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
            # Generate the design matrix of the B-spline basis
            bsp.get_matrix_B()
            if self.int_constraints is not None or self.pdf_constraint:
                bsp.get_matrices_S()
            bspline_bases.append(bsp)
        return bspline_bases

    def _fill_sp_args(self):

        """
        Fill the `sp_args` dictionary by default parameters on the case they are
        not provided.
        """

        if self.sp_args is None:
            self.sp_args = {}
        if self.sp_method == "grid_search":
            self.sp_args["grid"] = self.sp_args.get(
                "grid", [(0.01, 0.1, 1, 10) for _ in range(len(self.deg))]
            )
            self.sp_args["parallel"] = self.sp_args.get("parallel", False)
            self.sp_args["n_jobs"] = self.sp_args.get("n_jobs", -2)
            self.sp_args["top_n"] = self.sp_args.get("top_n", None)
        else:
            self.sp_args["verbose"] = self.sp_args.get("verbose", False)
            self.sp_args["x0"] = self.sp_args.get("x0", np.ones(len(self.deg)))
            self.sp_args["method"] = self.sp_args.get("method", "SLSQP")
            self.sp_args["bounds"] = self.sp_args.get(
                "bounds", [(1e-10, 1e16) for _ in range(len(self.deg))]
            )
        return None

    def _get_obj_func_arrays(self, y: np.ndarray) -> Dict[str, np.ndarray]:

        """
        Gather all the arrays used to define the objective function of the
        optimization funcion. These are the design matrices of the B-spline
        basis, the penalty matrices and the extended response variable sample
        (which is equal to zero outside the fitting region and coincides with
        the response variable sample inside it).

        Parameters
        ----------
        y : np.ndarray
            The response variable sample.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing design matrices of the B-spline basis, the
            penalty matrices and the extended response variable sample.
        """

        obj_matrices = {}
        obj_matrices["B"] = []
        obj_matrices["D"] = []
        obj_matrices["D_mul"] = []
        # The extended response variable sample dimensions can be obtained as
        # the number of rows of the design matrix B
        y_ext_dim = []
        for i, bsp in enumerate(self.bspline_bases):
            B = bsp.matrixB
            y_ext_dim.append(B.shape[0])
            obj_matrices["B"].append(B)
            penaltymat = PenaltyMatrix(bspline=bsp)
            P = penaltymat.get_penalty_matrix(**{"ord_d": self.ord_d[i]})
            obj_matrices["D"].append(penaltymat.matrixD)
            obj_matrices["D_mul"].append(P)

        # Reorder the response variable array so the covariate coordinates are
        # non-decreasing
        y_ext = np.zeros(tuple(y_ext_dim))
        y_ext[get_idx_fitting_region(self.bspline_bases)] = y
        obj_matrices["y"] = y_ext
        return obj_matrices

    def _initialize_model(
        self,
        obj_matrices: Dict[str, Iterable[np.ndarray]],
        data_normalizer: Optional[DataNormalizer] = None,
    ) -> mosek.fusion.Model:

        """
        Construct the optimization model.

        Parameters
        ----------
        lin_term : np.ndarray
            An array containing the coefficients of the linear term.
        L_B : np.ndarray
            The Cholesky decomposition of B.T @ B, where B is the Kronecker
            product of the B-spline basis matrices.
        L_D : Iterable[np.ndarray]
            An array containing the Cholesky decomposition of P_i, where P_i is
            the penalty matrix along the i axis.
        data_normalizer : Optional[DataNormalizer]
            The DataNormalizer object if `y_range` is not None and None
            otherwise. By default, None.

        Returns
        -------
        mosek.fusion.Model
            The optimization model.
        """

        M = mosek.fusion.Model()
        # Create the variables of the optimization problem
        mos_obj_f = ObjectiveFunction(bspline=self.bspline_bases, model=M)
        # For each axis, a smoothing parameter is needed
        sp = [M.parameter(f"sp_{i}", 1) for i, _ in enumerate(self.deg)]
        # Build the objective function of the problem
        mos_obj_f.create_obj_function(
            obj_matrices=obj_matrices, sp=sp, family=self.family
        )

        if self.pdf_constraint:
            if self.family.name != "gaussian":
                raise ValueError(
                    "Probability density function constraints are only implemented for Gaussian data."
                )
            pdf_cons = PDFConstraint(bspline=self.bspline_bases)
            # Incorporate the condition that the integral over all the space
            # must equal to 1
            pdf_cons.integrate_to_one(var_dict=mos_obj_f.var_dict, model=M)
            # Enforce the non-negativity constraint if it is not imposed
            # explicitly
            self.int_constraints = pdf_cons.nonneg_cons(self.int_constraints)

        if self.int_constraints is not None:
            max_deriv = max([max(v.keys()) for v in self.int_constraints.values()])
            if max_deriv > 1 and self.family.name != "gaussian":
                raise ValueError(
                    "Interval constraints are only implemented for non Gaussian data up to the first derivative "
                    f"Higher order derivative introduced in the constraints: {max_deriv})."
                )
            matrices_S = {i: bsp.matrices_S for i, bsp in enumerate(self.bspline_bases)}
            # Iterate for every variable with constraints and for every
            # derivative order
            for var_name in self.int_constraints.keys():
                for deriv in self.int_constraints[var_name].keys():
                    constraints = self.int_constraints[var_name][deriv]
                    if (
                        list(constraints.values())[0] != 0
                        and self.family.name != "gaussian"
                    ):
                        raise ValueError(
                            "No threshold is allowed in the shape constraints for non Gaussian data."
                        )
                    # Scale the integer constraints thresholds in the case the
                    # data is scaled
                    if data_normalizer is not None:
                        derivative = True if deriv != 0 else False
                        constraints = {
                            k: data_normalizer.transform(y=v, derivative=derivative)
                            for k, v in constraints.items()
                        }
                    matrices_S_copy = matrices_S.copy()
                    # Build the interval constraints
                    cons = IntConstraints(
                        bspline=self.bspline_bases,
                        var_name=var_name,
                        derivative=deriv,
                        constraints=constraints,
                    )
                    cons.interval_cons(
                        var_dict=mos_obj_f.var_dict, model=M, matrices_S=matrices_S_copy
                    )
        else:
            self.int_constraints = {}

        if self.pt_constraints is not None:
            if self.family.name != "gaussian":
                raise ValueError(
                    "Point constraints are only implemented for Gaussian data."
                )
            # Iterate for every combination of the derivative orders where
            # constraints must be enforced
            for deriv, info in self.pt_constraints.items():
                value = info[1]
                tolerance = info[2]
                # Scale the point constraints thresholds in the case the data is
                # scaled
                if data_normalizer is not None:
                    derivative = any(v != 0 for v in deriv)
                    value = data_normalizer.transform(y=value, derivative=derivative)
                    tolerance = data_normalizer.transform(
                        y=tolerance, derivative=False
                    ) - data_normalizer.transform(y=0, derivative=False)
                cons2 = PointConstraints(
                    pts=info[0],
                    value=value,
                    derivative=deriv,
                    bspline=self.bspline_bases,
                    tolerance=tolerance,
                )
                cons2.point_cons(var_dict=mos_obj_f.var_dict, model=M)
        else:
            self.pt_constraints = {}
        return M

    def _get_sp_grid_search(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:

        """
        Get the best smoothing parameter vector with the GCV minimizer criteria
        using grid search selection.

        Parameters
        ----------
        B_weighted : Iterable[np.ndarray]
            The weighted design matrix from the B-spline basis.
        Q_matrices : Iterable[np.ndarray]
            The array of matrices used in the GCV computation.
        y : np.ndarray
            The extended response variable sample.

        Returns
        -------
        Tuple[Union[int, float]]
            The best set of smoothing parameters
        """

        # Computes all the possible combinations for the smoothing parameters
        iter_sp = list(itertools.product(*self.sp_args["grid"]))
        # Run in parallel if the argument `parallel` is present
        if self.sp_args["parallel"] == True:
            gcv = Parallel(n_jobs=self.sp_args["n_jobs"])(
                delayed(GCV)(sp, obj_matrices, self.family) for sp in iter_sp
            )
        else:
            gcv = [
                GCV(
                    sp=sp,
                    obj_matrices=obj_matrices,
                    family=self.family,
                )
                for sp in iter_sp
            ]
        # Print the `top_n` combinations that minimizes the GCV
        if self.sp_args["top_n"] is not None:
            print_grid_search_results(
                x_val=iter_sp, obj_val=gcv, top_n=self.sp_args["top_n"]
            )

        return iter_sp[gcv.index(min(gcv))]

    def _get_sp_optimizer(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:

        """
        Get the best smoothing parameter vector with the GCV minimizer criteria
        using an optimizer from scipy.optimize.minimize.

        Parameters
        ----------
        B_weighted : Iterable[np.ndarray]
            The weighted design matrix from the B-spline basis.
        Q_matrices : Iterable[np.ndarray]
            The array of matrices used in the GCV computation.
        y : np.ndarray
            The extended response variable sample.

        Returns
        -------
        Tuple[Union[int, float]]
            The best set of smoothing parameters
        """

        # All argument in `sp_args` except `verbose` are arguments from
        # scipy.optimize.minimize, so create a copy without it
        scipy_optimize_params = self.sp_args.copy()
        scipy_optimize_params.pop("verbose", None)
        # Create a simulator to print the intermediate steps of the process if
        # "verbose" is active
        if self.sp_args["verbose"]:
            gcv_sim = Simulator(GCV)
        # Get the best set of smoothing parameters
        best_sp = scipy.optimize.minimize(
            gcv_sim.simulate if self.sp_args["verbose"] else GCV,
            args=(obj_matrices, self.family),
            callback=gcv_sim.callback if self.sp_args["verbose"] else None,
            **scipy_optimize_params,
        ).x
        return best_sp

    def _preprocessor(
        self, data: pd.DataFrame, y_col: str
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Preprocesses the input data, checking if it can be rearranged into a
        grid. If this is the case, the data is arranged accordingly.

        Parameters
        ----------
        data : pd.DataFrame
            Input data and target data.
        y_col : str
            The column name of the target variable.

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            The preprocessed data, either in scatter or grid format. The first
            element of the tuple corresponds to the covariate data and the
            second to the response data.
        """

        # Get the covariate column names
        x, y = scatter_to_grid(data=data, y_col=y_col)
        if len(data) == np.prod(y.shape) and np.isnan(y).sum() == 0:
            self.data_type = "gridded"
            logging.info("Data is rearranged into a grid.")
        else:
            self.data_type = "scattered"
            x = [row for row in data[data.columns.drop(y_col).tolist()].values.T]
            y = data[y_col].values
        return x, y

    def fit(
        self,
        data: pd.DataFrame,
        y_col: str,
        y_range: Optional[Iterable[Union[int, float]]] = None,
    ):

        """
        Compute the fitted decision variables of the B-spline expansion and the
        fitted values for the response variable.

        Parameters
        ----------
        data : pd.DataFrame
            Input data and target data.
        y_col : str
            The column name of the target variable.
        y_range : Optional[Iterable[Union[int, float]]]
            If not None, `y` is scaled in the range defined by this parameter.
            This scaling process is useful when `y` has very large norm, since
            MOSEK may not be able to find a solution in this case due to
            numerical issues. By default, None.

        Raises
        ------
        ValueError
            If the degree, difference order, number of interval and coordinates
            vectors length differs.
        ValueError
            If `sp_method` input is different from "grid_search" or "optimizer".
        NumericalError
            If MOSEK could not arrive to a feasible solution.
        """

        if len({len(i) for i in [self.deg, self.ord_d, self.n_int]}) != 1:
            raise ValueError("The lengths of `deg`, `ord_d`, `n_int` must agree.")

        if self.sp_method not in ["grid_search", "optimizer"]:
            raise ValueError(f"Invalid `sp_method`: {self.sp_method}.")

        x, y = self._preprocessor(data=data, y_col=y_col)

        # Construct the B-spline bases
        self.bspline_bases = self._get_bspline_bases(x=x)

        # Filling the arguments of the method used to determine the optimal set
        # of smoothing parameters
        _ = self._fill_sp_args()
        if y_range is not None:
            if self.family.name != "gaussian":
                raise ValueError(
                    "The argument `y_range` is only available for Gaussian data."
                )
            if len(y_range) != 2:
                raise ValueError("The range for `y` must be an interval.")
            data_normalizer = DataNormalizer(feature_range=y_range)
            _ = data_normalizer.fit(y)
            y = data_normalizer.transform(y)
        else:
            data_normalizer = None

        # Get the matrices used in the objective function
        obj_matrices = self._get_obj_func_arrays(y=y)

        # Auxiliary matrices derived from `obj_matrices`
        obj_matrices["B_w"] = get_weighted_B(bspline_bases=self.bspline_bases)
        obj_matrices["B_mul"] = list(map(matrix_by_transpose, obj_matrices["B_w"]))

        # Initialize the model
        M = self._initialize_model(
            obj_matrices=obj_matrices, data_normalizer=data_normalizer
        )
        model_params = {"theta": M.getVariable("theta")}
        for i, _ in enumerate(self.deg):
            model_params[f"sp_{i}"] = M.getParameter(f"sp_{i}")

        if self.sp_method == "grid_search":
            self.best_sp = self._get_sp_grid_search(
                obj_matrices=obj_matrices,
            )
        else:
            self.best_sp = self._get_sp_optimizer(
                obj_matrices=obj_matrices,
            )
        theta_shape = model_params["theta"].getShape()
        # Set the smoothing parameters vector as the optimal obtained in the
        # unconstrained setting
        for i, sp in enumerate(self.best_sp):
            model_params[f"sp_{i}"].setValue(sp)
        try:
            # Solve the problem
            with timer(
                tag=f"Solve the problem with smoothing parameters {tuple(self.best_sp)}: "
            ):
                M.solve()
            # Extract the fitted decision variables of the B-spline expansion
            self.sol = model_params["theta"].level().reshape(theta_shape)
            if y_range is not None:
                self.sol = data_normalizer.inverse_transform(y=self.sol)
            # Compute the fitted values of the response variable
            self.y_fitted = self.family.fitted(
                matrix_by_tensor_product([mat for mat in obj_matrices["B"]], self.sol)
            )
        except mosek.fusion.SolutionError as e:
            raise NumericalError(
                f"The solution for the smoothing parameter {self.best_sp} "
                f"could not be found due to numerical issues. The original error "
                f"was: {e}"
            )

        return None

    def predict(self, data: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        if data.empty:
            return np.array([])

        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        if self.data_type == "gridded":
            x = [np.unique(row) for row in data.values.T]
        else:
            x = [row for row in data.values.T]

        x_min = np.array([np.min(v) for v in x])
        x_max = np.array([np.max(v) for v in x])
        bsp_min = np.array([bsp.knots[bsp.deg] for bsp in self.bspline_bases])
        bsp_max = np.array([bsp.knots[-bsp.deg] for bsp in self.bspline_bases])
        # If some coordinates are outside the range where the B-spline bases
        # were defined, the problem must be fitted again
        if (x_min < bsp_min).sum() > 0:
            raise ValueError(
                f"Some of the coordinates are outside the definition range of "
                f"the B-spline bases."
            )
        if (x_max > bsp_max).sum() > 0:
            raise ValueError(
                f"Some of the coordinates are outside the definition range of "
                f"the B-spline bases."
            )
        # Compute the basis matrix at the coordinates to be predicted
        B_predict = [
            bsp.bspline_basis(x=x[i]) for i, bsp in enumerate(self.bspline_bases)
        ]
        if self.data_type == "gridded":
            y_pred = self.family.fitted(
                matrix_by_tensor_product([mat for mat in B_predict], self.sol)
            )
        else:
            raise ValueError(f"Not implemented.")
        return y_pred
