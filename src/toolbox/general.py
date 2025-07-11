# IMPORTS ######################################################################
from typing import Any, Iterable
import pandas as pd
from json import dumps
from numpy.random import shuffle
from gc import collect as gc_collect
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.cuda import is_available as cuda_available
import os
import numpy as np
from scipy.stats import norm
# SCRIPTS ######################################################################
def IdentityFunction(x : Any) -> Any: 
    """
    """
    return x

def pretty_printing_dictionnary(d : dict) -> str:
    """
    """
    return dumps(d, sort_keys = False, indent = 4)

def shuffle_list(l : list) -> list:
    """
    """
    shuffle(l)
    return l

def pretty_number(n : int, n_digits : int = 3) -> str :
    """
    """
    out = "0" * n_digits
    out += str(n)
    return out[-n_digits:]

def clean():
    """
    """
    empty_cache()
    if cuda_available():
        synchronize()
        ipc_collect()
    gc_collect()

def checkpoint_to_load(foldername : str, epoch : int) : 
    """
    """
    all_checkpoints : list[str] = [folder 
        for folder in os.listdir(foldername) if folder.startswith("checkpoint")]
    sorted_checkpoints : list[str] = sorted(all_checkpoints, 
        key = lambda file : int(file.split('-')[-1]))
    return sorted_checkpoints[epoch - 1]

def get_checkpoints(foldername : str) -> list[str]: 
    """
    """
    return [checkpoint_folder for checkpoint_folder in os.listdir(foldername) 
            if checkpoint_folder.startswith("checkpoint")]

def SUL_string(vec) : 
    """
    return a sorted list of unique string items
    """
    try : 
        # if the elements are strings representing floats
        sort = sorted(list(set(vec)), key = lambda x : float(str(x).lower()))
    except:
        # Else, sort alphabetically
        sort = sorted(list(set(vec)), key = lambda x : str(x).lower())
    return [str(x) for x in sort]

def get_band(vec : list[float], type : str, alpha : float = 0.9) -> float :
    """
    """
    mean = np.mean(vec)
    ## If vec == a * np.ones(n), returns an error.
    if np.equal(vec, mean).all(): 
        if type == "lower" : return 0
        elif type == "upper" : return 0
        else : return np.nan
    else : 
        band = norm.interval(alpha, loc=np.mean(vec), scale=np.std(vec))
        if type == "lower" : return mean - band[0]
        elif type == "upper" : return band[1] - mean
        else : return np.nan

def auto_log_range(vec_1, vec_2, window_frac : float = 0.1) -> tuple[float,float]: 
    """
    """
    min_vecs = np.log(min(min(vec_1), min(vec_2))) / np.log(10)
    max_vecs = np.log(max(max(vec_1), max(vec_2))) / np.log(10)
    print(min_vecs, max_vecs)
    return [min_vecs - window_frac, max_vecs + window_frac]

def get_uniques_values(vec_1, vec_2) -> list[float] : 
    """
    """
    return list(set([*vec_1, *vec_2]))

def get_most_frequent_item(vec : pd.Series) -> Any:
    """
    """
    return vec.mode().iloc[0]

def pretty_mean_and_ci(row : dict[str:float], precision : int = 3
    ) -> str: 
    """
    """
    ten_to_the_precision : str = pow(10,precision)
    M = int(row["mean"] * ten_to_the_precision) / ten_to_the_precision
    CI = int(row["upper_band"] * ten_to_the_precision) / ten_to_the_precision
    return f"{M}±{CI}"

def header_format(columns : list[str]) -> list[str] : 
    """
    """
    return [f"<b>{col}</b>" for col in columns]