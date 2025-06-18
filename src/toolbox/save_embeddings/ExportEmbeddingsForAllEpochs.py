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

    def export_all(self, device : str|None = None, 
        delete_files_after_routine : bool = False) -> None:
        """
        """
        for epoch in range(1, self.__n_epochs + 1) :
            ExportEmbeddingsForOneEpoch(self.__foldername, epoch, device).\
                routine(delete_files_after_routine)
    
    def routine(self, device : str|None = None, 
        delete_files_after_routine : bool = False) -> None: 
        self.export_all(device, delete_files_after_routine)