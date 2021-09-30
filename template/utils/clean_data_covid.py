import pandas as pd
import numpy as np
from typing import Union

from template.utils.normalize_data import normalize_data


def displaced_forecast_covid(
    xmax: Union[int, float],
    factor_disp: Union[int, float],
    factor_deriv: Union[int, float],
    deriv: np.ndarray,
):
    new_deriv = np.array([elem * factor_deriv for elem in deriv])
    x_pred = xmax + factor_disp * len(deriv)
    new_pts = np.arange(xmax + factor_disp, x_pred + factor_disp, factor_disp)
    return new_pts.tolist(), new_deriv.tolist(), x_pred


def filtered_covid_df(
    response_var,
    df=None,
    min_date=None,
    max_date=None,
    sexo=[],
    provincia=[],
):

    # Possible regions:

    # ['A', 'AB', 'AL', 'AV', 'B', 'BA', 'BI', 'BU', 'C', 'CA', 'CC',
    # 'CE', 'CO', 'CR', 'CS', 'CU', 'GC', 'GI', 'GR', 'GU', 'H', 'HU',
    # 'J', 'L', 'LE', 'LO', 'LU', 'M', 'MA', 'ML', 'MU', nan, 'NC', 'O',
    # 'OR', 'P', 'PM', 'PO', 'S', 'SA', 'SE', 'SG', 'SO', 'SS', 'T',
    # 'TE', 'TF', 'TO', 'V', 'VA', 'VI', 'Z', 'ZA']

    # Possible genders:

    # ['H', 'M']

    # Possible response variables

    # ['num_casos', 'num_hosp', 'num_uci', 'num_def']
    if df is None:
        df = pd.read_csv(
            r"https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv"
        )
    # Select the columns of interest from total df
    df = df.filter(items=["provincia_iso", "sexo", "grupo_edad", "fecha", response_var])
    # Select the correct gender and region
    if sexo:
        df = df.loc[df["sexo"].isin(sexo)]
    if provincia:
        df = df.loc[df["provincia_iso"].isin(provincia)]
    if len(sexo) != 1:
        df = (
            df.groupby(["grupo_edad", "fecha", "provincia_iso"])
            .agg({response_var: np.sum})
            .reset_index()
        )
    if len(provincia) != 1:
        df = (
            df.groupby(["grupo_edad", "fecha"])
            .agg({response_var: np.sum})
            .reset_index()
        )
    # Convert fecha to datetime
    df["fecha"] = pd.to_datetime(df["fecha"], infer_datetime_format=True)
    # Make the group ages a categorical variable
    df["grupo_edad"] = df["grupo_edad"].astype("category")
    # We remove the last day (problem from the site)
    if min_date is None:
        min_date = (df.groupby(["fecha"])[response_var].sum().cumsum() != 0).idxmax()
    mask_min_date = df["fecha"] >= min_date

    if max_date is None:
        max_date = df["fecha"].max()
    mask_max_date = df["fecha"] < max_date

    df = df[mask_max_date & mask_min_date]
    # Sort the dataframe by date and age group
    df = df.sort_values(by=["fecha", "grupo_edad"], ascending=[True, True])
    return df


def covid_agg_age(df, response_var, normalize=False):
    df = df.groupby(["fecha"]).agg({response_var: np.sum}).reset_index()
    df.set_index("fecha", inplace=True)
    if normalize:
        df = pd.DataFrame(
            data=normalize_data(df.values), index=df.index, columns=df.columns
        )
    return df


def covid_pivot_df(df, response_var, normalize=False):
    df = df.query("grupo_edad != 'NC'").pivot(
        index="fecha", columns="grupo_edad", values=response_var
    )
    if normalize:
        df = pd.DataFrame(
            data=normalize_data(df.values), index=df.index, columns=df.columns
        )

    return df


def get_days(df):
    return (
        (df.index.to_series().diff() / np.timedelta64(1, "D")).fillna(0).cumsum().values
    )
