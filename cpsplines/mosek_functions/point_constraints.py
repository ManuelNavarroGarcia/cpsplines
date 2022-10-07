from typing import Dict, Iterable, Tuple

import mosek.fusion
import numpy as np
import pandas as pd
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.utils.rearrange_data import scatter_to_grid


class PointConstraints:

    """
    Define the constraints that the smoothing (or a derivative in a particular
    direction) at a certain point must be equal, above or below a target value.
    The width of the bound with the sense "equalsTo" may be controlled by a
    tolerance parameter. All these constraints share the derivatives orders
    enforced.

    Parameters
    ----------
    derivative : Iterable[int]
        The orders of the derivatives. The first element corresponds to the
        derivative along the 0-th axis, the second element along the 1-st axis,
        and so on.
    sense : str
        It can be "greaterThan", "lessThan" or "equalsTo".
    bspline : Iterable[BsplineBasis]
        The B-spline bases objects.
    """

    def __init__(
        self,
        derivative: Iterable[int],
        sense: str,
        bspline: Iterable[BsplineBasis],
    ):
        self.derivative = derivative
        self.sense = sense
        self.bspline = bspline

    def point_cons(
        self,
        data: pd.DataFrame,
        y_col: str,
        data_arrangement: str,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
    ) -> Tuple[mosek.fusion.LinearConstraint]:

        """
        Constructs the point constraints for the fixed combination of
        derivative orders. These linear constraints are obtained evaluating the
        estimation of the B-splines expansion at the desired points (which yield
        to a linear term depending on elements from the multidimensional
        coefficient array) and setting the difference between this output and
        the value to be less than the tolerance.

        Parameters
        ----------
        data : pd.DataFrame
            Input data and target data.
        y_col : str
            The column name of the target variable.
        data_arrangement : str
            Type of arrangement of the data. It can be "gridded" or "scattered".
        var_dict : Dict[str, mosek.fusion.LinearVariable]
            A dictionary containing the decision variables.
        model : mosek.fusion.Model
            The MOSEK model of the problem.

        Returns
        -------
        Tuple[mosek.fusion.LinearConstraint]
            A tuple containing the point constraints.

        Raises
        ------
        ValueError
            If the array of coordinates `pts` and the derivative order vector
            `derivative` have different length than the number of B-spline bases.
        """

        x_cols = data.columns.drop([y_col, "tol"], errors="ignore").tolist()
        if len(x_cols) != len(self.bspline):
            raise ValueError(
                "The number of covariates and the derivative indexes must agree."
            )

        list_cons = []
        coef = []
        if data_arrangement == "gridded":
            x, y = scatter_to_grid(data=data, y_col=y_col)
            y = y.flatten().astype(float)
            # Get the evaluations of the coordinates at their respective B-spline
            # basis and the corresponding derivative order
            bsp_eval = [
                bsp.bspline_basis.derivative(nu=nu)(coords)
                for bsp, nu, coords in zip(self.bspline, self.derivative, x)
            ]
            # For every point constraint, extract the evaluation of the
            # corresponding coordinates and multiply them by the multidimensional
            # array of the expansion coefficients
            for i, _ in enumerate(y):
                bsp_x = [np.expand_dims(val[i, :], axis=1).T for val in bsp_eval]
                coef.append(
                    matrix_by_tensor_product_mosek(
                        matrices=bsp_x, mosek_var=var_dict["theta"]
                    )
                )
        if self.sense == "equalsTo":
            # The output should be constrained in (v - tol, v + tol) vstack is
            # necessary since `coef` may be a column matrix
            if "tol" in data.columns:
                right_side = mosek.fusion.Domain.inRange(
                    (y - data["tol"]).values, (y + data["tol"]).values
                )
            else:
                right_side = mosek.fusion.Domain.equalsTo(y)
        elif self.sense == "greaterThan":
            right_side = mosek.fusion.Domain.greaterThan(y)
        elif self.sense == "lessThan":
            right_side = mosek.fusion.Domain.lessThan(y)
        else:
            raise ValueError(f"The sense {self.sense} is not implemented.")
        list_cons.append(
            model.constraint(
                mosek.fusion.Expr.flatten(mosek.fusion.Expr.vstack(coef)),
                right_side,
            )
        )
        return tuple(list_cons)
