from functools import reduce
from typing import Iterable

import mosek.fusion
import numpy as np
from scipy.sparse import diags

from cpsplines.psplines.bspline_basis import BsplineBasis


class PDFConstraint:
    """
    Define the constraints that characterize a Probability Density Function,
    i.e., it must be non-negative and must integrate to one. For the first, only
    the dictionary of interval constraints is updated with the non-negative
    constraints (when they were not already included).

    Parameters
    ----------
    bspline : Iterable[BsplineBasis]
        An iterable containing the B-spline bases objects used to approximate
        the function to estimate.
    """

    def __init__(
        self,
        bspline: Iterable[BsplineBasis],
    ):
        self.bspline = bspline

    def integrate_to_one(
        self,
        var_dict: dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
    ) -> mosek.fusion.LinearConstraint:
        """
        Defines the constraint that the integral of the fitted hypersurface
        over the fitting and the prediction regions must be equal to one.

        Parameters
        ----------
        var_dict : Dict[str, mosek.fusion.LinearVariable]
            The dictionary that contains the decision variables used to define
            the objective function of the problem.
        model : mosek.fusion.Model
            The MOSEK model of the problem.

        Returns
        -------
        mosek.fusion.LinearConstraint
            The constraint that enforces to fitted hypersurface integrates to
            one.
        """

        banded_list = []
        for bsp in self.bspline:
            # For each B-spline basis, construct a Vandermonde matrix with only
            # the inner knots and with powers up to degree `deg` + 1. The first
            # column, containing ones, is dropped
            vander = np.vander(bsp.knots, N=bsp.deg + 2, increasing=True)[
                bsp.deg : -bsp.deg, 1:
            ]
            # The vector of constants obtained when integrating monomials
            integrand_coef = 1 / np.linspace(1, bsp.deg + 1, bsp.deg + 1)
            # Take the difference of a row minus the previous on the Vandermonde
            # matrix and multiply each column by the respective integration factor
            diff_mat = np.einsum("ij,j->ij", np.diff(vander, axis=0), integrand_coef)
            # Multiply each row by the corresponding matrix S and transpose, so
            # first term corresponds to first row and so on. Then, use this
            # vector to create a banded matrix with inner knots rows nd inner
            # rows plus degree columns
            banded = diags(
                np.array(
                    [np.dot(diff_mat[i], s) for i, s in enumerate(bsp.matrices_S)]
                ).T,
                range(bsp.deg + 1),
                shape=(bsp.matrixB.shape[1] - bsp.deg, bsp.matrixB.shape[1]),
            ).toarray()
            banded_list.append(banded)
        # Multiply by the coefficient array and sum all the entries
        sum_coef = mosek.fusion.Expr.sum(
            mosek.fusion.Expr.mul(
                reduce(np.kron, banded_list),
                mosek.fusion.Expr.flatten(var_dict["theta"]),
            )
        )
        cons = model.constraint(sum_coef, mosek.fusion.Domain.equalsTo(1.0))
        return cons

    def nonneg_cons(
        self,
        shape_constraints: dict[str, dict[int, dict[str, int | float]]] | None,
        feature_names: Iterable[str],
    ) -> dict[int, dict[int, dict[str, int | float]]]:
        """
        Includes non-negativity constraints to the problem if they were
        already not included.

        Parameters
        ----------
        shape_constraints : Optional[Dict[str, Dict[int, Dict[str, Union[int, float]]]]]
            The nested dictionary containing the interval constraints to be
            enforced.
        feature_names : Iterable[str]
            The name of the variables.

        Returns
        -------
        Dict[int, Dict[int, Dict[str, Union[int, float]]]]
            A interval constraints dictionary updated with non-negativity
            constraints along all axis, if they were not included explicitly.
        """

        # Create a dictionary if no interval constraints already exist
        if shape_constraints is None:
            shape_constraints = {}
        for col in feature_names:
            # Check if any interval constraints exist along each axis
            if shape_constraints.get(col, None) is not None:
                # Check if sign constraints exist for this axis
                if shape_constraints[col].get(0, None) is None:
                    # Update the constraint dictionary if non-negativity is
                    # missing
                    shape_constraints[col].update({0: {"+": 0}})
                else:
                    shape_constraints[col][0].update({"+": 0})
            # If no constraints are imposed on any axis, include non-negativity
            else:
                shape_constraints.update({col: {0: {"+": 0}}})
        return shape_constraints
