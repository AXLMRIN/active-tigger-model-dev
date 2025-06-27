# IMPORTS ######################################################################
import os
from .TestOneEpoch import TestOneEpoch
import pandas as pd
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class TestAllEpochs:
    """
    """
    def __init__(self, foldername, logger : CustomLogger) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__logger : CustomLogger = logger
        self.__n_epochs : int = len(
            [f for f in os.listdir(foldername) if f.startswith("checkpoint")])
        self.__results : list[dict] = []

    def run_tests(self, device : str|None = None, additional_tags : dict = {}):
        """
        """
        for epoch in range(1, self.__n_epochs + 1) :
            self.__results.append(
                TestOneEpoch(foldername = self.__foldername, epoch = epoch, 
                    logger = self.__logger, device = device).\
                    routine(additional_tags)
            )
    
    def save_results(self, filename : str):
        """
        """
        try : 
            df = pd.read_csv(f"{filename}")
            df = pd.concat((df, pd.DataFrame(self.__results)))
        except:
            df = pd.DataFrame(self.__results)
        finally:
            df.to_csv(f"{filename}", index = False)
    
    def routine(self, filename : str, device : str|None = None, 
        additional_tags : dict = {}):
        """
        """
        self.__logger(f"[TestAllEpochs] Routine start {self.__n_epochs} epochs---", 
            skip_line="before")
        
        self.run_tests(device,additional_tags)
        try:
            self.save_results(filename) 
        except Exception as e:
            raise ValueError(f"Test All Epochs failed saving.\n\nError:\n{e}")
        
        self.__logger("[TestAllEpochs] Routine finish ---", skip_line="after")