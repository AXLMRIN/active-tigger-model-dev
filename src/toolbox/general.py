# IMPORTS ######################################################################
from typing import Any
from json import dumps
from numpy.random import shuffle
# SCRIPTS ######################################################################
def IdentityFunction(x : Any) -> Any: 
    return x

def pretty_printing_dictionnary(d : dict) -> str:
    return dumps(d, sort_keys = False, indent = 4)

def shuffle_list(l : list) -> list:
    shuffle(l)
    return l

def pretty_number(n : int, n_digits : int = 3) -> str :
    out = "0" * n_digits
    out += str(n)
    return out[-n_digits:]