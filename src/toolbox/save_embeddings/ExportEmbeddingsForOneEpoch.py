# IMPORTS ######################################################################
import os

from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from torch import Tensor, no_grad, cat, save
from torch.cuda import is_available as cuda_available
from transformers import AutoModelForSequenceClassification

from ..general import checkpoint_to_load, clean
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class ExportEmbeddingsForOneEpoch: 
    """ExportEmbeddingsForOneEpoch is an object that loads a model after n epochs
    of training as well as data used during the training, then embeds the data 
    and labels to save under the pt format. It is meant to work with the files 
    created during the CustomTransformersPipeline routine.

    The main features and functions are:
        - Load a model after n epochs of training
        - Load encoded data and labels (train, eval and test sets)
        - Embed all
        - Save the embeddings.
        - Possibly deletes the heavy model files ("model.safetensors", 
            "optimizer.pt").

    For a given foldername, the embeddings and labels will be saved here:
        - foldername
            L checkpoint-XXX
            L ...
            L data
            L embeddings
                L epoch_XX
                    L train_labels.pt
                    L train_embeddings.pt
                    L test_labels.pt
                    L test_embeddings.pt
                L ...

    The routine function proceeds to all these steps.
    """
    def __init__(self, 
        foldername: str, 
        epoch : int, 
        logger : CustomLogger,
        device : str|None = None,
        batch_size : int = 64
        ) -> None:
        """Builds the ExportEmbeddingsForOneEpoch object.

        Parameters:
        -----------
            - foldername (str): full path to the checkpoints (saved during training)
                Equivalent to output_dir from CustomTransformersPipeline.
            - epoch (int): number of epochs of training, will choose what checkpoint
                to load. Epoch 0 = no training.
            - logger (CustomLogger): will give information as the data is processed.
            - device (str or None, default = None): device to load the model on.
                can be 'cpu', 'cuda' or 'cuda:X'.
            - batch_size (int, default = 64): the batch size used during the testing.

        Returns:
        --------
            /

        Inisialised variables:
        ----------------------
            DATA 
            - self.__ds (DatasetDict): dataset created during the step 1. Contains 
                at least a "test" split and 3 columns ('input_ids', 'attention_mask'
                and 'labels')

            MODEL
            - self.__foldername (str): full path to the checkpoints (saved during 
                training)Equivalent to output_dir from CustomTransformersPipeline.
            - self.__epoch (int): number of epochs of training, will choose what 
                checkpoint to load. Epoch 0 = no training.
            - self.__checkpoint (str): checkpoint in the folder corresponding to 
                the epoch we want to export.
            - self.__model (AutoModelForSequenceClassification): model loaded.
            - self.device (str or None, default = None): device to load the model on, 
                can be 'cpu', 'cuda' or 'cuda:X'.
            - self.__batch_size (int): the batch size used during the testing.

            COMMUNICATION AND SECURITY
            - self.__logger (CustomLogger): will give information as the data is 
                processed.
        """
        self.__foldername : str = foldername
        self.__epoch : str = epoch
        self.__logger : CustomLogger = logger
        self.__batch_size : int = batch_size
        self.__ds : DatasetDict = load_from_disk(f"{foldername}/data/")

        if device is None : 
            self.device = "cuda" if cuda_available() else "cpu"
        else :
            self.device = device

        self.__checkpoint : str = checkpoint_to_load(foldername, epoch)
        self.__model = AutoModelForSequenceClassification.\
            from_pretrained(f"{foldername}/{self.__checkpoint}").\
            to(device = self.device)
        
        if not(os.path.exists(f"{foldername}/embeddings/epoch_{epoch}")):
            os.makedirs(f"{foldername}/embeddings/epoch_{epoch}")
    
    def __get_embeddings(self, dataset : Dataset) -> tuple[Tensor, Tensor]:
        """Private function to generate the embeddings and the labels.

        Parameters:
        -----------
            - dataset (Dataset): data containing the encoded data to embed
                (columns : labels, input_ids, attention_mask)

        Returns:
        --------
            - Tensor : A Tensor for the labels 
            - Tensor : A Tensor for the embeddings 
        """
        with no_grad():
            embeddings = None
            labels = None

            for batch in dataset.batch(self.__batch_size, drop_last_batch=False): 
                batch_labels = Tensor(batch["labels"]).int().to(device=self.device)
                model_input = {
                    key : Tensor(batch[key]).int().to(device=self.device)
                    for key in ['input_ids', 'attention_mask']
                }
                batch_embeddings = self.__model.base_model(**model_input).\
                    last_hidden_state[:,0,:].squeeze()

                if embeddings is None: embeddings = batch_embeddings
                else :                 embeddings = cat((embeddings,batch_embeddings), 
                                                        axis = 0)

                if labels is None:  labels = batch_labels
                else :              labels = cat((labels,batch_labels), axis = 0)
        return labels, embeddings

    def export_train_embeddings(self) -> None:
        """Concatenate the train and eval dataset, call the __get_embeddings and 
        save the in the appropriate folder and name.

        Parameters:
        -----------
            /

        Returns:
        --------
            /
        """
        train_dataset : Dataset = concatenate_datasets(
            (self.__ds["train"], self.__ds["eval"]))
        labels, embeddings = self.__get_embeddings(train_dataset)
        save(labels, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/train_labels.pt")
        save(embeddings, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/train_embeddings.pt")
        
        # Logging
        self.__logger((f"(Epoch {self.__epoch}, checkpoint {self.__checkpoint}) "
                       "Train embeddings saved. "
                       f"Folder : {self.__foldername}/embeddings/epoch_{self.__epoch}/"))

    def export_test_embeddings(self):
        """Call the __get_embeddings and save the in the appropriate folder and 
        name.

        Parameters:
        -----------
            /

        Returns:
        --------
            /
        """
        labels, embeddings = self.__get_embeddings(self.__ds["test"])
        save(labels, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/test_labels.pt")
        save(embeddings, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/test_embeddings.pt")
        
        # Logging
        self.__logger((f"(Epoch {self.__epoch}, checkpoint {self.__checkpoint}) "
                       "Test embeddings saved. "
                       f"Folder : {self.__foldername}/embeddings/epoch_{self.__epoch}/"))
        
    def __delete_files(self) -> None:
        """Delete "model.safetensors" and "optimizer.pt" to save space.

        Parameters:
        -----------
            /

        Returns:
        --------
            /
        """
        # Logging
        self.__logger((f"(Epoch {self.__epoch}) WARNING: {self.__foldername}/"
                f"{self.__checkpoint}/(model.safetensors and optimizer.pt) will"
                " be permanently deleted."))
        for file in ["model.safetensors", "optimizer.pt"]:
            try : 
                os.remove(f"{self.__foldername}/{self.__checkpoint}/{file}")
            except Exception as e:
                print((f"File {self.__foldername}/{self.__checkpoint}/{file} "
                       f"could not be deleted because : \n{e}"))


    def routine(self, delete_files_after_routine : bool = False) -> None:
        """Routine used to load the models, embeds the encoded data and saves the
        files. If needed, deletes the heavy files.

        The error catching is very coarse and only helps narrow down where the 
        routine stopped. Needs an upgrade.
        
        Parameters:
        -----------
            - delete_files_after_routine(bool, default = False): boolean parameter
                to delete the heavy files of the model or not
        
        Returns:
        --------
            - None
        """
        self.export_train_embeddings()
        self.export_test_embeddings()
        del self.__ds, self.__model
        clean()
        if delete_files_after_routine:
            self.__delete_files()
    
