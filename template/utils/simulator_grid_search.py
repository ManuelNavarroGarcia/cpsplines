import pandas as pd


def print_grid_search_results(input_values, objective):
    df = pd.DataFrame(
        input_values, columns=[f"sp{i+1}" for i in range(len(input_values[0]))]
    )
    df.loc[:, "Objective"] = objective
    df = df.sort_values(by="Objective")
    print("Sorted GCV values")
    print(*df.columns, sep="\t\t")
    for i in range(df.shape[0]):
        print(*df.iloc[i, :], sep="\t\t")