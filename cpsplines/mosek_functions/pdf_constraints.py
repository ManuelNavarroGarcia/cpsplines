from typing import Dict, Iterable, Optional, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from scipy.sparse import diags


class PDFConstraint:

    """
    Define the constraints that ensures the non-negativity (or
    non-positivity) for the derivative order `derivative`along the respective
    axis of the variable `var_name`. For the unidimensional case, the
    requirements are completely fulfilled for every point in the range where the
    B-spline basis is defined. For the multivariate case, the constraints are
    imposed at the curves that pass through the inner knots of the B-spline
    basis. For all cases, the constraints are imposed following the Proposition
    1 in Bertsimas and Popescu (2002).

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
        var_dict: Dict[str, mosek.fusion.LinearVariable],
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
            # column, containing ones, is dropped too
            vander = np.vander(bsp.knots, N=bsp.deg + 2, increasing=True)[
                bsp.deg : -bsp.deg, 1:
            ]
            # The vector of constants obtained when integrating monomials
            integrand_coef = 1 / np.linspace(1, bsp.deg + 1, bsp.deg + 1)
            # Take the difference of a row minus the previous on the Vandermonde
            # matrix and multiply each column by the respective integration factor
            diff_mat = np.einsum("ij,j->ij", np.diff(vander, axis=0), integrand_coef)
            # The result of multiplying each row by the corresponding matrix S
            # is the same, so we take the product only on the first row. Then,
            # use this vector to create a banded matrix with inner knots rows
            # and inner rows plus degree columns
            banded = diags(
                np.dot(diff_mat[0], bsp.matrices_S[0]),
                range(bsp.deg + 1),
                shape=(bsp.matrixB.shape[1] - bsp.deg, bsp.matrixB.shape[1]),
            ).toarray()
            banded_list.append(banded)
        # Multiply by the coefficient array and sum all the entries
        sum_coef = mosek.fusion.Expr.sum(
            matrix_by_tensor_product_mosek(
                matrices=banded_list, mosek_var=var_dict["theta"]
            )
        )
        cons = model.constraint(sum_coef, mosek.fusion.Domain.equalsTo(1.0))
        return cons

    def nonneg_cons(
        self,
        int_constraints: Optional[Dict[int, Dict[int, Dict[str, Union[int, float]]]]],
    ) -> Dict[int, Dict[int, Dict[str, Union[int, float]]]]:

        """
        Includes non-negativity constraints to the problem if they were
        already not included.

        Parameters
        ----------
        int_constraints : Optional[Dict[int, Dict[int, Dict[str, Union[int, float]]]]]
            The nested dictionary containing the interval constraints to be
            enforced.

        Returns
        -------
        Dict[int, Dict[int, Dict[str, Union[int, float]]]]
            A interval constraints dictionary updated with non-negativity
            constraints along all axis, if they were not included explicitly.
        """

        # Create a dictionary if no interval constraints exist
        if int_constraints is None:
            int_constraints = {}
        for i in range(len(self.bspline)):
            # Check if any interval constraints exist along each axis
            if int_constraints.get(i, None) is not None:
                # Check if sign constraints exist for this axis
                if int_constraints[i].get(0, None) is None:
                    # Update the constraint dictionary if non-negativity is
                    # missing
                    int_constraints[i].update({0: {"+": 0}})
                else:
                    int_constraints[i][0].update({"+": 0})
            else:
                int_constraints.update({i: {0: {"+": 0}}})
        return int_constraints
