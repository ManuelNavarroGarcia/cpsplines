import numpy as np
from typing import Callable, Union


class Simulator:

    """
    Builds and solves the optimization problem to find the optimal coefficients of the B-splines
    basis for the case of one curve (one covariate, one response variable and one group). It is
    able to incorporate interval and point constraints, together with backwards and forward
    prediction.

    Attributes
    ----------
    func: Callable
        The objective function of the optimization problem.
    callback_count: int
        Number of times the callback is called.
    sol_eval: list
        A list that contains all the intermediate solutions used in the problem.
    func_eval: list
        A list that contains all the evaluations of the objective function of the problem.
    """

    def __init__(self, function):
        self._func = function
        self._callback_count = 0
        self._sol_eval = []
        self._func_eval = []

    @property
    def func(self):
        return self._func

    @property
    def callback_count(self):
        return self._callback_count

    @property
    def sol_eval(self):
        return self._sol_eval

    @property
    def func_eval(self):
        return self._func_eval

    @func.setter
    def func(self, func: Callable):
        self._func = func

    @callback_count.setter
    def callback_count(self, callback_count: int):
        self._callback_count = callback_count

    @sol_eval.setter
    def sol_eval(self, sol_eval: list):
        self._sol_eval = sol_eval

    @func_eval.setter
    def func_eval(self, func_eval: list):
        self._func_eval = func_eval

    def simulate(self, x_k, *args) -> Union[int, float]:

        """
        Executes the actual simulation and returns the result, while
        updating the lists too. Pass to optimizer without arguments or
        parentheses.

        Inputs
        ------
        x_k: list
            The actual value for the solution.

        Returns
        -------
        (Union[int, float]) The objective function evaluated at x_k.

        """

        result = self.func(x_k, *args)
        self.sol_eval.append(x_k)
        self.func_eval.append(result)
        return result

    def callback(self, x_k, *_):
        """
        Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses.

        Inputs
        ------
        x_k: list
            The actual value for the solution.

        """

        # Locate the position in sol_eval that coincides with x_k. Once this
        # position is found, break the loop to save the position.
        for i, x in reversed(list(enumerate(self.sol_eval))):
            if np.allclose(x, x_k):
                break

        sp_info = ""
        # For each component in the actual solution, generate an string containing its
        # value and finally the value of the objective function stored in func_eval
        for comp in x_k:
            sp_info += f"{comp:10.5e}\t"
        sp_info += f"{self.func_eval[i]:10.5e}"

        # Set the title of the callback, with the smoothing parameter names and the
        # objective function label
        if not self.callback_count:
            title_list = [f"sp{j+1}" for j, _ in enumerate(x_k)] + ["Objective"]
            print("Starting Gradient Descent algorithm")
            print(*title_list, sep="\t\t")
        # Print the actual solution and the actual objective function value
        print(sp_info)
        self.callback_count += 1