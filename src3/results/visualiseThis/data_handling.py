# === === === === === === === === === === === === === === === === === === === ==
# IMPORTS
# === === === === === === === === === === === === === === === === === === === ==
import pandas as pd
import numpy as np
from typing import Any
from . import ROOT
from scipy.stats import t

# === === === === === === === === === === === === === === === === === === === ==
# CLASS
# === === === === === === === === === === === === === === === === === === === ==
class genData:
    def __init__(self, 
            filenames : str|list[str]|dict[str,str]|None = None,
            existing_dataframe : pd.DataFrame|None = None, 
            concat_col : str|None = None
        ) -> None: 
        self.__df = None
        concat_col = "filename_csv" if concat_col is None else concat_col
        if  (filenames is not None)&\
            (existing_dataframe is None): 
            if isinstance(filenames,str):
                    try : self.__df = pd.read_csv(filenames)
                    except Exception: print(Exception)

            elif isinstance(filenames, list):
                for file in filenames : 
                    try : 
                        loop_df = pd.read_csv(f"{ROOT}/{file}")
                        loop_df[concat_col] = [file] * len(loop_df)
                        self.__concat_to_df(loop_df)
                    except Exception as e: print(e); pass

            elif isinstance(filenames, dict):
                for key, file in filenames.items(): 
                    try : 
                        loop_df = pd.read_csv(f"{ROOT}/{file}")
                        loop_df[concat_col] = [key] * len(loop_df) 
                        self.__concat_to_df(loop_df)
                    except Exception as e: print(e); pass
            # Add a model, lr columns : 
            self.__df["model"] = self.\
                __df["filename"].\
                apply(lambda x : '-'.join(x.split("-")[3:-3]))
            self.__df["lr"] = self.\
                __df["filename"].\
                apply(lambda x : '-'.join(x.split("-")[-3:-1])).\
                astype(float)
            
        elif    (filenames is None)&\
                (existing_dataframe is not None)&\
                isinstance(existing_dataframe, pd.DataFrame): 
            self.__df = existing_dataframe
        else : print("WARNING : your object is empty")
                
    def columns(self) -> list[str] : return self.__df.columns.to_list()

    def __repr__(self) -> str: return self.__df.__str__()
    
    def rename(self, **kwargs) -> None:
        self.__df.rename(**kwargs, inplace = True)

    def __concat_to_df(self, new_df):
        if self.__df is None : self.__df = new_df
        else : self.__df = pd.concat((self.__df, new_df))

    def pick(self, 
            columns : list[str], 
            indexes : list[int]|None = None): 
        if indexes is None : 
            return genData(existing_dataframe = self.__df.loc[:,columns])
        else : 
            return genData(existing_dataframe = self.__df.loc[indexes,columns])
    
    def to_DataFrame(self) -> pd.DataFrame : 
        return self.__df

    def __getitem__(self, key, filter):
        return self.__df.loc[:,key].to_list()
    
    def get_mean_and_half_band(self, 
            groupbyColumns : list[str]|str, 
            column : str = "f1_macro",
            new_columns : dict|None = None,
            N : int|None = None, 
            alpha : float = 0.9
        ) -> pd.DataFrame : 
        """Evaluate the mean and the half band of a vector in a sub dataframe
        grouped by the groupbyColumns list of strings.
        Possible to rename the new_columns"""
        result =  self.__df.groupby(groupbyColumns, as_index=False).agg(
            f1_macro_mean = (column, lambda col : mean_over_N_bests(col, N)),
            CI_f1_macro_lower = (column,lambda col:half_band_over_N_bests(col,N,alpha)[0]),
            CI_f1_macro_upper = (column,lambda col:half_band_over_N_bests(col,N,alpha)[1])
        )
        if new_columns is not None: 
            result.rename(mapper = new_columns, inplace = True)
        
        return result

# === === === === === === === === === === === === === === === === === === === ==
# FUNCTIONS
# === === === === === === === === === === === === === === === === === === === ==                    
def mean_over_N_bests(col : list,N : int|None = None) -> float:
    if N is not None : N = min(N, len(col))
    else : N = len(col)
    col = sorted(col,reverse=True)[:N]
    return np.mean(col)

def half_band_over_N_bests(
        col : list, 
        N : int|None = None, 
        alpha : float = 0.9
    ) -> float:

    if N is None : N = len(col)
    else : N = min(N, len(col))
    col = sorted(col,reverse=True)[:N]

    sigma = np.std(col, ddof = 1)
    t_crit = t.ppf(q = alpha, df = N)

    M = np.mean(col)
    EM = t_crit * sigma #/ np.sqrt(N)
    return M + EM, M - EM

