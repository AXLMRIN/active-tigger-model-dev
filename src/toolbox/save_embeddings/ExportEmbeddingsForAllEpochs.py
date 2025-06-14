# IMPORTS ######################################################################
import os
from .ExportEmbeddingsForOneEpoch import ExportEmbeddingsForOneEpoch
# SCRIPTS ######################################################################
class ExportEmbeddingsForAllEpochs:
    """
    """
    def __init__(self, foldername) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__n_epochs : int = len(
            [f for f in os.listdir(foldername) if f.startswith("checkpoint")])
        self.__results : list[dict] = []

    def export_all(self, device : str|None = None):
        """
        """
        for epoch in range(self.__n_epochs) : 
            ExportEmbeddingsForOneEpoch(self.__foldername, epoch, device).\
                routine()
    
    def routine(self, device : str|None = None) -> None: 
        self.export_all(device)