# IMPORTS ######################################################################
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForSequenceClassification
from torch import Tensor, no_grad, cat, save
from torch.cuda import is_available as cuda_available
from ..general import checkpoint_to_load, clean
import os
# SCRIPTS ######################################################################
class ExportEmbeddingsForOneEpoch: 
    """
    """
    def __init__(self, foldername: str, epoch : int, 
        device : str|None = None) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__epoch : str = epoch
        self.__ds : DatasetDict = load_from_disk(f"{foldername}/data/")

        if device is None : 
            self.device = "cuda" if cuda_available() else "cpu"
        else :
            self.device = device

        self.__checkpoint : str = checkpoint_to_load(foldername, epoch)
        self.__model = AutoModelForSequenceClassification.\
            from_pretrained(f"{foldername}/{self.__checkpoint}")
        
        if not(os.path.exists(f"{foldername}/embeddings/epoch_{epoch}")):
            os.makedirs(f"{foldername}/embeddings/epoch_{epoch}")
    
    def __get_embeddings(self, dataset : Dataset) -> tuple[Tensor, Tensor]:
        with no_grad():
            embeddings = None
            labels = None

            for batch in dataset.batch(16): 
                batch_labels = Tensor(batch["labels"]).int().to(device=self.device)
                model_input = {
                    key : Tensor(batch[key]).int().to(device=self.device)
                    for key in ['input_ids', 'attention_mask']
                }
                batch_embeddings = self.__model.base_model(**model_input).\
                    last_hidden_state[:,0,:].squeeze()

                if embeddings is None:
                    embeddings = batch_embeddings
                else :
                    embeddings = cat((embeddings,batch_embeddings), axis = 0)

                if labels is None:
                    labels = batch_labels
                else :
                    labels = cat((labels,batch_labels), axis = 0)
        return labels, embeddings

    def export_train_embeddings(self):
        """
        """
        train_dataset : Dataset = concatenate_datasets(
            (self.__ds["train"], self.__ds["eval"]))
        labels, embeddings = self.__get_embeddings(train_dataset)
        save(labels, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/train_labels.pt")
        save(embeddings, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/train_embeddings.pt")

    def export_test_embeddings(self):
        """
        """
        labels, embeddings = self.__get_embeddings(self.__ds["test"])
        save(labels, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/test_labels.pt")
        save(embeddings, f"{self.__foldername}/embeddings/epoch_{self.__epoch}/test_embeddings.pt")

    def __delete_files(self) -> None:
        """
        """
        print((f"WARNING: {self.__foldername}/{self.__checkpoint}/"
               "(model.safetensors and optimizer.pt) are permanently deleted."))
        for file in ["model.safetensors", "optimizer.pt"]:
            try : 
                os.remove(f"{self.__foldername}/{self.__checkpoint}/{file}")
            except Exception as e:
                print((f"File {self.__foldername}/{self.__checkpoint}/{file} "
                       "could not be deleted because : \n{e}"))


    def routine(self, delete_files_after_routine : bool = False) -> None:
        self.export_train_embeddings()
        self.export_test_embeddings()
        del self.__ds, self.__model
        clean()
        if delete_files_after_routine:
            self.__delete_files()
    
