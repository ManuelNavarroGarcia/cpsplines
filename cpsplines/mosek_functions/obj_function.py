from functools import reduce
from typing import Dict, Iterable, Tuple, Union

import mosek.fusion
import numpy as np
import statsmodels.genmod.families.family
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.utils.cholesky_semidefinite import cholesky_semidef
from cpsplines.utils.fast_kron import (
    fast_kronecker_product,
    matrix_by_tensor_product,
    penalization_term,
)


class ObjectiveFunction:

    """
    Define the objective function of the optimization problem. The initial
    convex quadratically optimization problem is reformulated as an equivalent
    conic quadratic problem with linear objective function.

    Parameters
    ----------
    bspline : Iterable[BsplineBasis]
        An iterable containing the B-spline bases objects used to approximate
        the function to estimate.
    model : mosek.fusion.Model
        The MOSEK model of the problem.

    Attributes
    ----------
    var_dict: Dict[str, mosek.fusion.LinearVariable]
        The resulting variable dictionary from the method `_create_var_dict`.
    """

    def __init__(self, bspline: Iterable[BsplineBasis], model: mosek.fusion.Model):
        self.bspline = bspline
        self.model = model

    def _create_var_dict(self, n: int) -> Dict[str, mosek.fusion.LinearVariable]:

        """
        Creates the variables of the optimization problem. These variables are:

        - The coefficient multidimensional array ("theta") with dimensions
          k_1 x ... x k_N, where k_i is the cardinal of B-spline basis along the
          i-th dimension.
        - The artificial variable of the design matrix product ("t_B") included
          in the reformulation of the quadratic term.
        - The artificial variable of the i-th penalty matrix product ("t_D_i")
          when restating this summand as a rotated quadratic cone constraint.

        Parameters
        ----------
        n : int
            Number of smoothing parameters needed in the optimization function.

        Returns
        -------
        Dict[str, mosek.fusion.LinearVariable]
            A dictionary containing the variables involved on the objective
            function of the optimization problem. The keys of the dictionary are
            the names of the variables and the values are the variables
            themselves.
        """

        var_dict = {}
        # The coefficient multidimensional array
        variable_shape = [bsp.matrixB.shape[1] for bsp in self.bspline]
        var_dict["theta"] = self.model.variable(
            "theta",
            variable_shape,
            mosek.fusion.Domain.unbounded(),
        )
        for i in range(n):
            var_dict[f"t_D_{i}"] = self.model.variable(
                f"t_D_{i}", 1, mosek.fusion.Domain.greaterThan(0.0)
            )
        return var_dict

    def create_obj_function(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
        sp: Iterable[Union[int, float]],
        family: statsmodels.genmod.families.family,
    ) -> Tuple[Union[None, mosek.fusion.ConicConstraint]]:

        """
        Creates the objective function to be minimized. Although the penalized
        sum of squares is directly stated as a quadratic optimization function,
        the positive semidefiniteness of the matrices in the quadratic term
        enables to reformulate it as a linear objective function together with
        rotated quadratic cone constraints.

        Indeed, each of the individual summands x^TAx in the quadratic term of
        the objective function may be replaced with a new artificial variable t
        and impose the constraint x^TAx <= t, which has an equivalent
        characterization as the (k+2)-vector (t, 1/2, L^Tx) belonging to the
        (k+2)-dimensional rotated quadratic cone, where L satisfies A = LL^T.

        Parameters
        ----------
        obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
            A dictionary containing the necessary arrays (the basis matrices,
            the penalty matrices and the response variable sample) used to
            construct the objective function.
        sp : Iterable[Union[int, float]]
            An iterable containing the smoothing parameters.
        family : statsmodels.genmod.families.family
            The specific exponential family distribution where the response
            variable belongs to.

        References
        ----------
        - Alizadeh, F., & Goldfarb, D. (2003). Second-order cone programming.
          Mathematical programming, 95(1), 3-51.

        Returns
        -------
        Tuple[Union[None, mosek.fusion.ConicConstraint]]
            The linear objective function and the rotated quadratic cone
            constraints resulting from reformulating the summands on the
            quadratic term.

        Raises
        ------
        ValueError
            If lengths of the smoothing parameter vector and penalty matrix
            iterable differ.
        """

        L_D = penalization_term(matrices=obj_matrices["D"])

        if len(sp) != len(L_D):
            raise ValueError(
                "The number of smoothing parameters and penalty matrices must agree."
            )

        # Generate the decision variables involved in the objective function
        self.var_dict = self._create_var_dict(n=len(sp))

        # Linear term coefficients must be dot multiplied by the row major
        # vectorization of the basis expansion multidimensional array
        # coefficients
        flatten_theta = mosek.fusion.Var.flatten(self.var_dict["theta"])

        cons = []
        # Create the rotated quadratic cone constraint of each D^TD
        for i, L in enumerate(L_D):
            cons.append(
                self.model.constraint(
                    f"rot_cone_{i}",
                    mosek.fusion.Expr.vstack(
                        self.var_dict[f"t_D_{i}"],
                        1,
                        mosek.fusion.Expr.mul(
                            mosek.fusion.Matrix.sparse(L), flatten_theta
                        ),
                    ),
                    mosek.fusion.Domain.inRotatedQCone(),
                )
            )

        # The rotated cone reformulation on the penalty term yield summands on
        # the objective function of the form sp*t_D, where t_D is the new
        # artificial variable introduced in the characterization
        obj = mosek.fusion.Expr.add(
            [
                mosek.fusion.Expr.dot(s, self.var_dict[f"t_D_{i}"])
                for i, s in enumerate(sp)
            ]
        )

        if family.name == "gaussian":
            self.var_dict |= {
                "t_B": self.model.variable(
                    "t_B", 1, mosek.fusion.Domain.greaterThan(0.0)
                )
            }
            # Compute the linear term coefficients of the objective function
            lin_term = matrix_by_tensor_product(
                [mat.T for mat in obj_matrices["B_w"]], obj_matrices["y"]
            ).flatten()

            # Compute the Cholesky decompositions (A = L @ L.T)
            L_B = reduce(
                fast_kronecker_product,
                list(map(cholesky_semidef, obj_matrices["B_mul"])),
            )
            # Create the rotated quadratic cone constraint of B^TB
            cons.append(
                self.model.constraint(
                    "rot_cone_B",
                    mosek.fusion.Expr.vstack(
                        self.var_dict["t_B"],
                        1,
                        mosek.fusion.Expr.mul(
                            mosek.fusion.Matrix.sparse(L_B.T), flatten_theta
                        ),
                    ),
                    mosek.fusion.Domain.inRotatedQCone(),
                )
            )
            # The rotated cone reformulation on the basis term yield a summand of
            # the artificial variable t_B included during the reformulation and
            # a linear term depending on the response variable sample
            obj = mosek.fusion.Expr.add(
                mosek.fusion.Expr.sub(
                    self.var_dict["t_B"], mosek.fusion.Expr.dot(lin_term, flatten_theta)
                ),
                obj,
            )
        elif family.name == "poisson":
            self.var_dict |= {
                "t": self.model.variable(
                    "t",
                    np.prod(obj_matrices["y"].shape),
                    mosek.fusion.Domain.greaterThan(0.0),
                )
            }

            lin_term = matrix_by_tensor_product(
                [mat.T for mat in obj_matrices["B_w"]], obj_matrices["y"]
            ).flatten()

            coef = mosek.fusion.Expr.flatten(
                matrix_by_tensor_product_mosek(
                    matrices=obj_matrices["B_w"], mosek_var=self.var_dict["theta"]
                )
            )
            cons.append(
                self.model.constraint(
                    mosek.fusion.Expr.hstack(
                        self.var_dict["t"],
                        mosek.fusion.Expr.constTerm(
                            np.prod(obj_matrices["y"].shape), 1.0
                        ),
                        coef,
                    ),
                    mosek.fusion.Domain.inPExpCone(),
                )
            )

            obj = mosek.fusion.Expr.sub(
                mosek.fusion.Expr.add(
                    mosek.fusion.Expr.sum(self.var_dict["t"]),
                    obj,
                ),
                mosek.fusion.Expr.dot(lin_term, flatten_theta),
            )

        # Generate the minimization objective function object
        obj = self.model.objective(
            "obj",
            mosek.fusion.ObjectiveSense.Minimize,
            obj,
        )
        return tuple([obj] + cons)
