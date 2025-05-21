import pandas as pd
import numpy as np
from typing import Any
from . import ROOT

class genData:
    def __init__(self, filenames : str|list[str]|dict[str,str]): 
        self.__df = None
        if isinstance(filenames,str):
                try : self.__df = pd.read_csv(filenames)
                except Exception: print(Exception)

        elif isinstance(filenames, list):
            for file in filenames : 
                try : 
                    loop_df = pd.read_csv(f"{ROOT}/{file}")
                    loop_df["filename_csv"] = [file] * len(loop_df)
                    self.__concat_to_df(loop_df)
                except Exception as e: print(e); pass

        elif isinstance(filenames, dict):
            for key, file in filenames.items(): 
                try : 
                    loop_df = pd.read_csv(f"{ROOT}/{file}")
                    loop_df["filename_csv"] = [key] * len(loop_df)
                    self.__concat_to_df(loop_df)
                except Exception as e: print(e); pass
                
    def columns(self) -> list[str] : return self.__df.columns.to_list()
    def __repr__(self) -> str: return self.__df.__str__()
        
    def __concat_to_df(self, new_df):
        if self.__df is None : self.__df = new_df
        else : self.__df = pd.concat((self.__df, new_df))

    def get_columns(self, 
        columns : list[str], 
        indexes : list[int]|None = None
        ) -> np.ndarray: 
        if indexes is None : return self.__df.loc[:,columns].to_numpy()
        else : return self.__df.loc[indexes,columns].to_numpy()

            
                    
