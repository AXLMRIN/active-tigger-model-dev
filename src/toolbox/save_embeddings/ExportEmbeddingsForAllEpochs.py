# IMPORTS ######################################################################
import os
from ..CustomLogger import CustomLogger
from .ExportEmbeddingsForOneEpoch import ExportEmbeddingsForOneEpoch
# SCRIPTS ######################################################################
class ExportEmbeddingsForAllEpochs:
    """
    """
    def __init__(self, foldername : str, logger : CustomLogger) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__logger = logger
        self.__n_epochs : int = len(
            [f for f in os.listdir(foldername) if f.startswith("checkpoint")])

    def export_all(self, device : str|None = None, 
        delete_files_after_routine : bool = False) -> None:
        """
        """
        for epoch in range(1, self.__n_epochs + 1) :
            ExportEmbeddingsForOneEpoch(foldername = self.__foldername, 
                    epoch = epoch, logger = self.__logger, device = device).\
                routine(delete_files_after_routine)
    
    def routine(self, device : str|None = None, 
        delete_files_after_routine : bool = False) -> None: 
        """
        """
        self.__logger((f"[ExportEmbeddingsForAllEpochs] Routine start "
                       f"({self.__n_epochs} epochs) ---"), skip_line="before")
        
        self.export_all(device, delete_files_after_routine)
        
        self.__logger(f"[ExportEmbeddingsForAllEpochs] Routine finish ---", 
            skip_line="after")