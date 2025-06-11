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
        new_df["lr"] = new_df["filename"].apply(lambda x : float("-".join(x.split("-")[-3:-1])))
        if df is None: df = new_df
        else: df = pd.concat((df, new_df))

    return df
    
def SUL_string(vec) : 
    '''return a sorted list of unique string items'''
    return sorted(list(set(vec)), key = lambda x : str(x).lower())

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

def estimate_time_to_process(df : pd.DataFrame):
    header = f"{'Method':^30}|{'q05':^15}|{'q50':^15}|{'q95 ':^15}|{'Total':<15}"
    hline = "-" * len(header)
    print(header)
    print(hline)
    for method, sub_df in df.groupby("method"):
        col = sub_df["time"].dropna().to_list()
        if len(col) == 0 : 
            print((f"{method:^30}|"
                f"{'NaN':^15}|"
                f"{'NaN':^15}|"
                f"{'NaN':^15}|"
                f"{'NaN':<15}"))
        else : 
            sorted_col = sorted(col)
            N05 = int(0.05 * len(sorted_col)); 
            q05 = sorted_col[N05]; q05MIN = q05//60; q05S = q05 - q05MIN * 60
            N50 = int(0.50 * len(sorted_col)); 
            q50 = sorted_col[N50]; q50MIN = q50//60; q50S = q50 - q50MIN * 60
            N95 = int(0.95 * len(sorted_col)); 
            q95 = sorted_col[N95]; q95MIN = q95//60; q95S = q95 - q95MIN * 60; 
            totH = sum(sorted_col) // 3600; totMIN = round((sum(sorted_col) - totH * 3600)/60)
            print((f"{method:^30}|"
                f"{'%i min %.1f s'%(q05MIN, q05S):^15}|"
                f"{'%i min %.1f s'%(q50MIN, q50S):^15}|"
                f"{'%i min %.1f s'%(q95MIN, q95S):^15}|"
                f"{'%i h %i min'%(totH, totMIN):<15}"))
    print(hline)

def onlyBestEpoch(df : pd.DataFrame) -> pd.DataFrame:
    new_df = []
    for _, sub_df  in df.groupby(["model","lr","method","n_samples","iteration"]):
        bestF1idx = np.argmax(sub_df["f1_macro"])
        new_df.append(sub_df.iloc[bestF1idx])

    return pd.DataFrame(new_df)

def f1_HF_vs_f1_all(df : pd.DataFrame) :
    # keys : model, method, f1_HF, f1_macro
    new_df = None

    for (model, lr, epoch), sub_df in df.groupby(["model", "lr", "epoch"]):
        f1_HF = 0
        if epoch == 0 : f1_HF = 0.33
        else: f1_HF = float(sub_df.loc[sub_df["method"] == "MLPClassifier (HF)", "f1_macro"].iloc[0])
        sub_df["f1_HF"] = [f1_HF] * len(sub_df)

        if new_df is None: new_df = sub_df
        else : new_df = pd.concat((new_df, sub_df))
    
    return new_df