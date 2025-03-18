# TODO write a banner
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import Dataset
import numpy as np
from pandas import read_csv
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer
)
from transformers.tokenization_utils_base import BatchEncoding
# Custom
from .AutoClassifierRoutineConfig import AutoClassifierRoutineConfig
from .general import storage_options, split_test_train_valid, compute_metrics

# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class AutoClassifierRoutine: 
    def __init__(self, config : AutoClassifierRoutineConfig) -> None:
        self.config = config

    def open_file(self):
        # Open file :
        try : 
            self.ds : Dataset = Dataset.from_pandas(read_csv(
                self.config.filename_open_s3, storage_options = storage_options()
            ))
            print("Dataset loaded with s3")
        except:
            self.ds : Dataset = Dataset.from_pandas(read_csv(
                self.config.filename_open_local
            ))
            print("Dataset loaded locally")

        # Retrieve the labels
        self.label : list[str] = list(set(self.ds[self.config.label_col])); 
        self.n_labels : int = len(self.label)
        self.id2label : dict[int:str] = {i : cat for i,cat in enumerate(self.label)}
        self.label2id : dict[str:int] = {cat:i for i,cat in enumerate(self.label)}
        print("Categories : " + ", ".join([cat for cat in self.label]),"\n")
        print(">>> Data loading - Done")
    
    def __preprocess_function(self,batch_of_rows : dict) -> dict:
        """To make sure the format will fit with the model expectations,
        we destroy the current dataset and rebuild one from scratch
        """
        batch_of_rows_out : dict = {}
        batch_of_rows_out["sentence"] = [
            sentence.lower() for sentence in batch_of_rows[self.config.sentence_col]
        ]
        for label_col in self.label2id : 
            batch_of_rows_out[label_col] =[
                label_col == label for label in batch_of_rows[self.config.label_col] 
            ]
        return batch_of_rows_out

    def preprocess_data(self) -> None : 
        self.ds = self.ds.map(
            lambda batch_of_rows : self.__preprocess_function(batch_of_rows), 
            batched = True, batch_size = self.config.batch_size
        )
        # The ds should only have n_label + 1 columns, ie one column for each 
        # label (list of bool) and one column (sentence) for the sentence (list of str)
        print(">>> Preprocess - Done")

    def split_ds(self) -> None:
        #Split the dataset in test, train, and valid dataset
        self.ds = split_test_train_valid(self.ds)

    def __encoding_data_function(self, batch_of_rows : dict) -> dict:
        # Comes from a tutorial
        encoding : BatchEncoding = self.tokenizer(
            batch_of_rows["sentence"], **self.config.tokenizer_settings 
        )
        labels_matrix = np.zeros((len(batch_of_rows["sentence"]), self.n_labels))
        for label in self.label2id:
            labels_matrix[:,self.label2id[label]] = batch_of_rows[label]

        # Associate the metadata to the encodings
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def tokenize_data(self) -> None:
        # Tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        print(">>> Tokenizer loading - Done")

        self.encoded_dataset = self.ds.map(
            lambda batch_of_rows : self.__encoding_data_function(batch_of_rows),
            batched = True, batch_size = self.config.batch_size
        )
        print(">>> Tokenization - Done")

    def load_model(self) -> None :
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            problem_type = "multi_label_classification", 
            num_labels = self.n_labels,
            id2label = self.id2label, 
            label2id = self.label2id
        ).to(self.config.device)

        if self.config.only_train_classifier : 
            print(("\n"
                "=============\n"
                "WARNING : You are only training the classifier, the embedding"
                "model is frozen\n"
                "=============\n"
            ))
            for name, param in self.model.named_parameters():
                if name.startswith("classifier") : param.requires_grad = True
                else : param.requires_grad = False
        print(">>> Model loading - Done")
    
    def __subsetting_ds(self) -> None:
        print(("\n"
            "=============\n"
            "WARNING for dev purposes you are only using a subset of the "
            "dataset you loaded\n"
            "=============\n"
        ))
        self.encoded_dataset["train"] = self.encoded_dataset["train"].\
                                            select(range(0,20))
        self.encoded_dataset["validation"] = self.encoded_dataset["validation"].\
            select(range(0,20))

    def train(self):
        trainer = Trainer(
            self.model,
            self.config.training_args,
            train_dataset = self.encoded_dataset["train"],
            eval_dataset = self.encoded_dataset["validation"],
            processing_class = self.tokenizer,
            compute_metrics = compute_metrics
        )
        trainer.train()

    def run(self):
        self.open_file()
        self.preprocess_data()
        self.split_ds()
        self.tokenize_data()
        if self.config.dev_mode : self.__subsetting_ds()
        self.load_model()
        self.train()

