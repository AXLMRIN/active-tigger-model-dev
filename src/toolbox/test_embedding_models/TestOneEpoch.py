# IMPORTS ######################################################################
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch import Tensor, load
from sklearn.metrics import f1_score
import numpy as np
from torch.cuda import is_available as cuda_available
from ..general import checkpoint_to_load, clean
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class TestOneEpoch: 
    """
    """
    # UPGRADE make possible to change the measure
    def __init__(self, foldername: str, epoch : int, logger : CustomLogger,
        device : str|None = None) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__epoch : str = epoch
        self.__logger : CustomLogger = logger
        self.__ds : DatasetDict = load_from_disk(f"{foldername}/data/")

        if device is None : 
            self.device = "cuda" if cuda_available() else "cpu"
        else :
            self.device = device

        checkpoint : str = checkpoint_to_load(foldername, epoch)
        self.__model = AutoModelForSequenceClassification.\
            from_pretrained(f"{foldername}/{checkpoint}").\
            to(device = self.device)

        self.__training_args : TrainingArguments = load(
            f"{foldername}/{checkpoint}/training_args.bin", weights_only=False)
        
        with open(f"{foldername}/model_name.txt", "r") as file:
            self.__model_name : str = file.read()

        self.__measure : str = "f1_macro" #UPGRADE


        (self.__score) = (None, ) * 1

    def run_test(self):
        """
        """
        labels_true : list[int] = []
        labels_pred : list[int] = []

        for batch in self.__ds["test"].batch(4, drop_last_batch=False):
            model_input = {
                'input_ids' : Tensor(batch['input_ids']).\
                                to(dtype = int).to(device=self.device), 
                'attention_mask' : Tensor(batch['attention_mask']).\
                                to(dtype = int).to(device=self.device) 
            }

            logits : np.ndarray = self.__model(**model_input).logits.\
                detach().cpu().numpy()
            
            batch_of_true_label : list[int] = [
                np.argmax(row).item() for row in batch["labels"]]
            labels_true.extend(batch_of_true_label)

            batch_of_pred_label : list[int] = [
                np.argmax(row).item() for row in logits]
            labels_pred.extend(batch_of_pred_label)
            
        self.__score = f1_score(labels_true, labels_pred, average='macro')

        # Logging
        self.__logger(f"(Epoch {self.__epoch}) Testing - Done (score : {self.__score})")
        
    def return_result(self, additional_tags : dict = {}) -> dict:
        """
        """
        return {
            "folder" : self.__foldername,
            "epoch" : int(self.__epoch) + 1,
            "score" : self.__score, 
            "measure" : self.__measure, 
            "learning_rate" : self.__training_args.learning_rate,
            "optim" : self.__training_args.optim,
            "warmup_ratio" : self.__training_args.warmup_ratio,
            "weight_decay" : self.__training_args.weight_decay,
            "embedding_model" : self.__model_name,
            **additional_tags
        }

    def routine(self, additional_tags : dict = {}) -> dict:
        """
        """
        try: 
            self.run_test()
        except Exception as e:
            raise ValueError(f"Test One Epoch could not run the test.\n\nError:\n{e}")
        output = self.return_result(additional_tags)
        del self.__model, self.__ds
        clean()
        return output
    
