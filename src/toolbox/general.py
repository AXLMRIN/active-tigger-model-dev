# IMPORTS ######################################################################
from typing import Any
from json import dumps
from numpy.random import shuffle
from gc import collect as gc_collect
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.cuda import is_available as cuda_available
import os
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