import pandas as pd
from typing import Iterable, Union


def print_grid_search_results(
    x_val: Iterable[Iterable[Union[int, float]]],
    obj_val: Iterable[Union[int, float]],
    top_n: int,
) -> None:
    """
    Prints the best `top_n` combinations of values that minimizes the
    objective function value.

    Parameters
    ----------
    x_val : Iterable[Iterable[Union[int, float]]]
        An iterable containing the arrays used to evaluate the objective
        function.
    obj_val : Iterable[Union[int, float]]
        The objective function values, with the same order as `x_val`
    top_n : int
        The number of top results to be shown.
    """
    df = pd.DataFrame(x_val, columns=[f"sp{i+1}" for i, _ in enumerate(x_val[0])])
    df.loc[:, "Objective"] = obj_val
    df = df.sort_values(by="Objective")
    print(f"Top {top_n} combinations minimizing the GCV criterion")
    print(*df.columns, sep="\t\t")
    for i in range(min(top_n, df.shape[0])):
        print(*df.iloc[i, :], sep="\t\t")
    return None
