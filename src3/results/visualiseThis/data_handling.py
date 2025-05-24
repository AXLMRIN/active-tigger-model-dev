# === === === === === === === === === === === === === === === === === === === ==
# IMPORTS
# === === === === === === === === === === === === === === === === === === === ==
import pandas as pd
import numpy as np
from typing import Any
from . import ROOT
from scipy.stats import norm
# === === === === === === === === === === === === === === === === === === === ==
# FUNCTIONS
# === === === === === === === === === === === === === === === === === === === ==

def fetch_data(filename_dictionnary : dict[str:str], usecols : list[str]) : 
    df : pd.DataFrame = None
    for name, filename in filename_dictionnary.items():
        new_df = pd.read_csv(f"{ROOT}/{filename}", usecols=usecols)
        new_df["method"] = [name] * len(new_df)
        new_df["model"] = new_df["filename"].apply(lambda x : "-".join(x.split("-")[3:-3]))
        new_df["lr"] = new_df["filename"].apply(lambda x : "-".join(x.split("-")[-3:-1]))
        if df is None: df = new_df

        print(len(new_df), len(df))

    return df
    


def mean_over_N_bests(col : list) -> float:
    return np.mean(col)

def half_band_over_N_bests(
        col : list, 
        alpha : float = 0.9
    ) -> float:

    return norm.interval(alpha, loc = np.mean(col), scale = np.std(col))

def concat_to_df(concat_df, new_df):
    if concat_df is None : return new_df
    else : return pd.concat((concat_df, new_df))

def get_mean_and_half_band(
        df : pd.DataFrame,
        groupbyColumns : list[str]|str, 
        column : str = "f1_macro",
        N : int|None = None, 
        alpha : float = 0.9
    ) -> pd.DataFrame : 
    
    out : list[dict] = []
    for keyValues, sub_df in df.groupby(groupbyColumns):
        col = sub_df[column].to_list()
        if N is not None : 
            local_N = min(len(col), N)
            col = sorted(col, reverse=True)[:local_N]
        mean = np.mean(col)
        band = norm.interval(alpha, loc=mean, scale=np.std(col))

        out.append({
            **{key : value for key, value in zip(groupbyColumns, keyValues)},
            'f1_macro_mean' : mean,
            'CI_f1_macro_lower' : band[0],
            'CI_f1_macro_upper' : band[1]
        })

    return pd.DataFrame(out)