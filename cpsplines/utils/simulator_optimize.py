import numpy as np
from typing import Callable, Iterable, Union


class Simulator:

    """
    A callback function to display the current value of the arguments and the
    value of the objective function at each iteration. Based on the class
    developed in https://stackoverflow.com/a/59330005/4983192

    Parameters
    ----------
    func: Callable
        The objective function to be optimized.

    Attributes
    ----------
    callback_count: int
        Number of times the callback is called.
    sol_eval: list
        A list that contains all the intermediate solutions used in the problem.
    func_eval: list
        A list that contains all the evaluations of the objective function of the problem.
    """

    def __init__(self, function: Callable):
        self.func = function
        self.callback_count = 0
        self.sol_eval = []
        self.func_eval = []

    def simulate(self, x_k: Iterable[Union[int, float]], *args) -> Union[int, float]:

        """
        Executes the actual simulation and returns the result, while updating
        the attributes `sol_eval` and `func_eval`. This must be passed to the
        optimizer without arguments or parentheses.

        Inputs
        ------
        x_k: Iterable[Union[int, float]]
            The actual value for the solution.

        Returns
        -------
        (Union[int, float]) The objective function evaluated at x_k.
        """

        result = self.func(x_k, *args)
        self.sol_eval.append(x_k)
        self.func_eval.append(result)
        return result

    def callback(self, x_k: Iterable[Union[int, float]], *_) -> None:
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
            print("Starting the optimization algorithm")
            print(*title_list, sep="\t\t")
        # Print the actual solution and the actual objective function value
        print(sp_info)
        self.callback_count += 1
        return None
