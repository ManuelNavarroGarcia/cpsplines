from typing import Dict, Iterable, Tuple, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis


class PointConstraints:

    """
    Define the constraints that the smoothing (or a derivative in a particular
    direction) at a certain point must be bounded around an input value. The
    width of the bound is controlled by a tolerance parameter. All these
    constraints share the derivatives orders enforced.

    Parameters
    ----------
    pts : Iterable[np.ndarray]
        An iterable of arrays containing the coordinates of the points where the
        constraints are enforced. The first vector contain the coordinates in
        the 0-th axis, the second vector the coordinates in the 1-st axis, and
        so on.
    value : mosek.fusion.Model
        The middle point of the bounds that the smoothing (or derivative) must
        fulfill.
    derivative : Iterable[int]
        The orders of the derivatives. The first element corresponds to the
        derivative along the 0-th axis, the second element along the 1-st axis,
        and so on.
    bspline : Iterable[BsplineBasis]
        The B-spline bases objects.
    tolerance : Union[int, float]
        The tolerance used to define the bounds.
    """

    def __init__(
        self,
        pts: Iterable[np.ndarray],
        value: Iterable[Union[int, float]],
        derivative: Iterable[int],
        bspline: Iterable[BsplineBasis],
        tolerance: Union[int, float],
    ):
        self.pts = pts
        self.value = value
        self.derivative = derivative
        self.bspline = bspline
        self.tolerance = tolerance

    def point_cons(
        self,
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

        if any(len(x) != len(self.bspline) for x in [self.pts, self.derivative]):
            raise ValueError(
                "`pts` and `derivative` lengths must be equal the number of covariates."
            )
        # Get the evaluations of the coordinates at their respective B-spline
        # basis and the corresponding derivative order
        bsp_eval = {
            i: bsp.bspline_basis.derivative(nu=self.derivative[i])(self.pts[i])
            for i, bsp in enumerate(self.bspline)
        }

        list_cons = []
        # For every point constraint, extract the evaluation of the
        # corresponding coordinates and multiply them by the multidimensional
        # array of the expansion coefficients
        for i, v in enumerate(self.value):
            bsp_x = [np.expand_dims(val[i, :], axis=1).T for val in bsp_eval.values()]
            coef = matrix_by_tensor_product_mosek(
                matrices=bsp_x, mosek_var=var_dict["theta"]
            )
            # The output should be constrained on the interval (v - tol, v + tol)
            list_cons.append(
                model.constraint(
                    coef,
                    mosek.fusion.Domain.greaterThan(v - self.tolerance),
                )
            )
            list_cons.append(
                model.constraint(
                    coef,
                    mosek.fusion.Domain.lessThan(v + self.tolerance),
                )
            )
        return tuple(list_cons)
