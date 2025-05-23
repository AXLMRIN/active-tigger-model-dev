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

def fetch_data(
        filenames : str|list[str]|dict[str,str]|None = None,
        existing_dataframe : pd.DataFrame|None = None, 
        concat_col : str|None = None,
        usecols : list[str]|None = None
    ) -> None: 
    
    df = None

    concat_col = "filename_csv" if concat_col is None else concat_col

    if  (filenames is not None)&\
        (existing_dataframe is None): 
        if isinstance(filenames,str):
                try : df = pd.read_csv(filenames, usecols = usecols)
                except Exception: print(Exception)

        elif isinstance(filenames, list):
            for file in filenames : 
                try : 
                    loop_df = pd.read_csv(f"{ROOT}/{file}", usecols = usecols)
                    loop_df[concat_col] = [file] * len(loop_df)
                    df = concat_to_df(df, loop_df)
                except Exception as e: print(e); pass

        elif isinstance(filenames, dict):
            for key, file in filenames.items(): 
                try : 
                    loop_df = pd.read_csv(f"{ROOT}/{file}", usecols = usecols)
                    loop_df[concat_col] = [key] * len(loop_df) 
                    df = concat_to_df(df, loop_df)
                except Exception as e: print(e); pass

        # Add a model, lr columns : 
        df["model"] = df["filename"].\
            apply(lambda x : '-'.join(x.split("-")[3:-3]))
        df["lr"] = df["filename"].\
            apply(lambda x : '-'.join(x.split("-")[-3:-1])).\
            astype(float)
        
    elif    (filenames is None)&\
            (existing_dataframe is not None)&\
            isinstance(existing_dataframe, pd.DataFrame): 
        df = existing_dataframe
    
    else : print("WARNING : your object is empty")
    # Reset the index : 
    df.index = [i for i in range(len(df))]

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