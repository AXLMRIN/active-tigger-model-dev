# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import load_dataset, Dataset, DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
# Native

# Custom
from .Config import Config
from .general import (
    split_test_train_valid,
    split
)

# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class CustomDataset:
    #TODO Add logger
    #TODO implement saving strategy
    def __init__(self, config : Config) -> None:
        self.config = config
        self.ds : None | DatasetDict = None
        self.open_status : bool = False
        self.preprocess_status : bool = False
        self.labels : None | list[str] = None
        self.n_labels : None | int = None

    def open_dataset(self) -> None:
        #UPGRADE make possible to load from the internet
        #UPGRADE for now, only open csv but need to make it possible for different format
        if self.config.dataset_filename.endswith(".csv") : 
            ds_config : pd.DataFrame = pd.\
                read_csv(self.config.dataset_filename).\
                rename(
                    {
                        self.config.dataset_label_col : "label",
                        self.config.dataset_text_col : "text"
                    }, 
                    axis = 1)

            # NOTE This method is deprecated
            # Split dataset into 3 Datasets (train, test, validation)
            # self.ds = split_test_train_valid(ds_config["whole"],
            #             **self.config.dataset_split_parameters_DEPRECATED
            # )
            
            # Split dataset into 3 Datasets (train, test, validation)
            self.ds = split(ds_config, **self.config.dataset_split_parameters)
            self.open_status = True
            
    def find_labels(self) -> None: 
        # Look for the labels
        self.labels = list(set(self.ds["train"]["label"]))
        self.n_labels = len(self.labels)
        self.config.dataset_label2id = {label : i for i,label in enumerate(self.labels)}
        self.config.dataset_id2label = {i : label for i,label in enumerate(self.labels)}
        # Save to config to be used elsewhere
        self.config.dataset_n_labels = self.n_labels

        # evaluate the number of elements per class in the training set
        instances = self.ds["train"].to_pandas().groupby("label").size()
        # Change the weights for the loss function
        self.config.model_loss_parameters["weight"] = Tensor([1 for i in range(self.n_labels)])
        for label in self.config.dataset_label2id : 
            self.config.model_loss_parameters["weight"]\
                [self.config.dataset_label2id[label]] = max(instances) / float(instances[label])    
    
    def preprocess_data(self,preprocess_function_text, 
        preprocess_function_label)->None:

        for split in tqdm(["train", "test", "validation"], desc = "Preprocess",
                            leave = False, position = 1): 
            loader = DataLoader(self.ds[split])
            new_ds = {
                "text" : [],
                "label" : []
            }
            for batch in tqdm(loader, desc = split, leave = False, position = 2): 
                new_ds["text"].extend(preprocess_function_text(batch))
                new_ds["label"].extend(preprocess_function_label(batch))
            
            self.ds[split] = Dataset.from_dict(new_ds)
        self.preprocess_status = True

    def debug(self):
        for split in ["train", "test", "validation"] : 
            self.ds[split] = self.ds[split].select(range(0,20))

    def __str__(self):
        return (
            "Custom Dataset object\n"
            "Status :\n"
            f"\t- File open : {'Done' if self.open_status else ''}\n"
            f"\t- labels : {self.labels}\n"
            f"\t- Preprocess : {'Done' if self.preprocess_status else ''}\n"
            f"\t- sizes : \n"
            f"\t\t- train : {len(self.ds['train'])}\n"
            f"\t\t- validation : {len(self.ds['validation'])}\n"
            f"\t\t- test : {len(self.ds['test'])}\n"
        )