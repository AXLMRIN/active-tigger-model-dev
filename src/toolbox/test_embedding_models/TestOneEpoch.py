# IMPORTS ######################################################################
from typing import Any

from datasets import load_from_disk, Dataset, DatasetDict
import numpy as np
from sklearn.metrics import f1_score
from torch import Tensor, load
from torch.cuda import is_available as cuda_available
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from ..general import checkpoint_to_load, clean
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class TestOneEpoch: 
    """TestOneEpoch is an object that loads a model after n epochs of training 
    and evaluate it's performances. It meant to work with the files created during 
    the CustomTransformersPipeline routine.

    The main features and functions are:
        - load a model after n epochs of training.
        - load encoded data and labels (test set)
        - classify entries and evaluate performances
        - return a dictionnary resuming the performances and metadata describing
        the training.

    The routine function proceeds to all these steps.   
    """
    # UPGRADE make possible to change the measure
    def __init__(self, foldername: str, epoch : int, logger : CustomLogger,
        device : str|None = None, batch_size : int = 64) -> None:
        """Builds the TestOneEpoch object.

        Possible UPDATE: change the measure to use.

        Parameters:
        -----------
            - foldername (str): full path to the checkpoints (saved during training)
                Equivalent to output_dir from CustomTransformersPipeline.
            - epoch (int): number of epochs of training, will choose what checkpoint
                to load. Epoch 0 = no training.
            - logger (CustomLogger): will give information as the data is processed.
            - device (str or None, default = None): device to load the model on, 
                can be 'cpu', 'cuda' or 'cuda:X'.
            - batch_size (int): the batch size used during the testing.

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
            - self.__model (AutoModelForSequenceClassification): model loaded.
            - self.__batch_size (int): the batch size used during the testing.
            
            METADATA 
            - self.__training_args (TrainingArguments): training arguments used 
                during the training to feed the metadata.
            - self.__model_name (str): name of the model used to feed the metadata.
            - self.__metric (str): the metric used to evaluate the performance of
                the model. For now only f1_macro available. The metric will be 
                returned in the metadata.
            - self.__score (float): the score of the model after testing.

            COMMUNICATION AND SECURITY
            - self.__logger (CustomLogger): will give information as the data is 
                processed.
            
        """
        self.__foldername : str = foldername
        self.__epoch : str = epoch
        self.__logger : CustomLogger = logger
        self.__ds : DatasetDict = load_from_disk(f"{foldername}/data/")
        self.__batch_size : int = batch_size

        if device is None : 
            self.device = "cuda" if cuda_available() else "cpu"
        else :
            self.device = device

        # Load the training args to retrieve return afterwards.
        self.__training_args : TrainingArguments = load(
            f"{foldername}/{checkpoint}/training_args.bin", weights_only=False)
        
        # Load the model
        with open(f"{foldername}/model_name.txt", "r") as file:
            self.__model_name : str = file.read()

        # Choose what checkpoint to load, and load the model
        checkpoint : str = checkpoint_to_load(foldername, epoch)
        self.__model = AutoModelForSequenceClassification.\
            from_pretrained(f"{foldername}/{checkpoint}").\
            to(device = self.device)

        self.__metric : str = "f1_macro"

        (self.__score) = (None, ) * 1

    def run_test(self):
        """In batch, predicts the label and save the true label and then evaluate
        the score

        Parameters:
        -----------
            /
        
        Returns:
        --------
        """
        labels_true : list[int] = []
        labels_pred : list[int] = []

        for batch in self.__ds["test"].batch(self.__batch_size, drop_last_batch=False):
            model_input = {
                'input_ids' : Tensor(batch['input_ids']).\
                                to(dtype = int).\
                                to(device=self.device), 
                'attention_mask' : Tensor(batch['attention_mask']).\
                                to(dtype = int).\
                                to(device=self.device) 
            }

            logits : np.ndarray = self.__model(**model_input).logits.\
                detach().cpu().numpy()
            
            batch_of_true_label : list[int] = [
                np.argmax(row).item() for row in batch["labels"]]
            labels_true.extend(batch_of_true_label)

            batch_of_pred_label : list[int] = [
                np.argmax(row).item() for row in logits]
            labels_pred.extend(batch_of_pred_label)
        
        # Evaluate performance
        self.__score = f1_score(labels_true, labels_pred, average='macro')

        # Logging
        self.__logger(f"(Epoch {self.__epoch}) Testing - Done (score : {self.__score})")
        
    def return_result(self, additional_tags : dict[str:Any] = {}) -> dict:
        """Builds a dictionnary mixing the results and metadata from the training
        routine. Allows for extra tags.

        Parameters:
        -----------
            - additional_tags(dict[str:Any]): Allows to include additionnal tags.
        
        Returns:
        --------
            - dict[str:Any]
        """
        return {
            "folder" : self.__foldername,
            "epoch" : int(self.__epoch) + 1,
            "score" : self.__score, 
            "measure" : self.__metric, 
            "learning_rate" : self.__training_args.learning_rate,
            "optim" : self.__training_args.optim,
            "warmup_ratio" : self.__training_args.warmup_ratio,
            "weight_decay" : self.__training_args.weight_decay,
            "embedding_model" : self.__model_name,
            **additional_tags
        }

    def routine(self, additional_tags : dict = {}) -> dict:
        """Routine used to load the models, run the testing and return the 
        metadata. 

        The error catching is very coarse and only helps narrow down where the 
        routine stopped. Needs an upgrade.
        
        Parameters:
        -----------
            - additional_tags(dict[str:Any]): Allows to include additionnal tags.
        
        Returns:
        --------
            - dict[str:Any]
        """
        try: 
            self.run_test()
        except Exception as e:
            raise ValueError(f"Test One Epoch could not run the test.\n\nError:\n{e}")
        output = self.return_result(additional_tags)
        del self.__model, self.__ds
        clean()
        return output
    
