# IMPORTS ######################################################################
import os
from ..CustomLogger import CustomLogger
from .ExportEmbeddingsForOneEpoch import ExportEmbeddingsForOneEpoch
# SCRIPTS ######################################################################
class ExportEmbeddingsForAllEpochs:
    """ExportEmbeddingsForAllEpochs is an object calling ExportEmbeddingsForOneEpoch
    for all epochs in a folder

    The main features and functions:
        - Evaluates the number of epochs for a given training
        - Run the ExportEmbeddingsForOneEpoch routine for all epochs

    The routine function proceeds to all these steps.   
    """
    def __init__(self, 
        foldername : str, 
        logger : CustomLogger,
        device : str|None = None,
        batch_size : int = 64
        ) -> None:
        """Builds the ExportEmbeddingsForAllEpochs object.

        Parameters:
        -----------
            - foldername (str): full path to the checkpoints (saved during training)
                Equivalent to output_dir from CustomTransformersPipeline.
            - logger (CustomLogger): will give information as the data is processed.
            - device (str or None, default = None): device to load the model on.
                can be 'cpu', 'cuda' or 'cuda:X'.
            - batch_size (int, default = None): the batch size used during the 
                testing.

        Returns:
        --------
            /

        Initialised variables:
        ----------------------
            - self.__foldername (str): full path to the checkpoints (saved during 
                training)Equivalent to output_dir from CustomTransformersPipeline.
            - self.__n_epochs (int): number of epochs during the training.
            - self.__logger (CustomLogger): will give information as the data is 
                processed.
            - self.__device (str or None, default = None): the batch size used
                during the testing.
            - self.__batch_size (int): the batch size used during the testing.
        """
        self.__foldername : str = foldername
        self.__logger = logger
        self.__n_epochs : int = len(
            [f for f in os.listdir(foldername) if f.startswith("checkpoint")])
        self.__batch_size : int = batch_size
        self.__device : str|None = device

    def export_all(self, 
        delete_files_after_routine : bool = False
        ) -> None:
        """Run the TestOneEpoch routine for all epochs.

        Parameters:
        -----------
            - delete_files_after_routine(bool, default = False): boolean parameter
                to delete the heavy files of the model or not

        Returns:
        --------
            /
        """
        for epoch in range(1, self.__n_epochs + 1) :
            ExportEmbeddingsForOneEpoch(
                foldername = self.__foldername, 
                epoch = epoch, 
                logger = self.__logger, 
                device = self.__device,
                batch_size = self.__batch_size
            ).\
                routine(delete_files_after_routine)
    
    def routine(self,
        delete_files_after_routine : bool = False
        ) -> None: 
        """Run the TestOneEpoch routine for all epochs.

        Parameters:
        -----------
            - delete_files_after_routine(bool, default = False): boolean parameter
                to delete the heavy files of the model or not

        Returns:
        --------
            /
        """
        self.__logger((f"[ExportEmbeddingsForAllEpochs] Routine start "
                       f"({self.__n_epochs} epochs) ---"), skip_line="before")
        
        self.export_all(delete_files_after_routine)
        
        self.__logger(f"[ExportEmbeddingsForAllEpochs] Routine finish ---", 
            skip_line="after")