# TODO write a banner
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import Dataset, DatasetDict
import numpy as np
from pandas import read_csv, DataFrame
from torch.utils.data import DataLoader
from torch.cuda import synchronize, ipc_collect, empty_cache
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer
)
from transformers.tokenization_utils_base import BatchEncoding
from datetime import datetime
from tqdm import tqdm
# Native
from logging import getLogger
from time import time
import gc
# Custom
from .AutoClassifierRoutineConfig import AutoClassifierRoutineConfig
from .general import (
    storage_options, split_test_train_valid, compute_metrics,multi_label_metrics
)

# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class AutoClassifierRoutine: 
    def __init__(self, config : AutoClassifierRoutineConfig) -> None:
        self.config = config
        self.logger = getLogger("GENERAL_LOGGER")
        self.model = None
        self.ds = None
        self.tokenizer = None

    def open_file(self):
        # Open file :
        start = time()
        try : 
            self.ds : Dataset = Dataset.from_pandas(read_csv(
                self.config.filename_open_s3, storage_options = storage_options()
            ))
            self.logger.info("Dataset loaded with s3")
        except:
            self.ds : Dataset = Dataset.from_pandas(read_csv(
                self.config.filename_open_local
            ))
            self.logger.info("Dataset loaded locally")
        end = time()
        # Retrieve the labels
        self.label : list[str] = list(set(self.ds[self.config.label_col])); 
        self.n_labels : int = len(self.label)
        self.id2label : dict[int:str] = {i : cat for i,cat in enumerate(self.label)}
        self.label2id : dict[str:int] = {cat:i for i,cat in enumerate(self.label)}
        print("Categories : " + ", ".join([cat for cat in self.label]),"\n")
        self.logger.info(f">>> Data loading - Done ({end - start :.2f})")
    
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
        print("Start preprocess")# TODELETE
        dataloader = DataLoader(self.ds, 
            batch_size = self.config.preprocess_batch_size, shuffle = True
        )
        new_ds = {}
        start = time()
        for batch in tqdm(dataloader, desc = "Preprocess data"): 
            batch_result = self.__preprocess_function(batch)

            for key in batch_result : 
                if key not in new_ds.keys(): new_ds[key] = []
                new_ds[key].extend(batch_result[key])
        end = time()
        self.ds = Dataset.from_dict(new_ds)
        # The ds should only have n_label + 1 columns, ie one column for each 
        # label (list of bool) and one column (sentence) for the sentence (list of str)
        self.logger.info(f">>> Preprocess - Done ({end - start :.2f})")

        del dataloader, batch, batch_result, new_ds
        gc.collect()

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
        start = time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        end = time()
        self.logger.info(f">>> Tokenizer loading - Done ({end - start :.2f})")

        encoded_dataset = {}
        start = time()
        for split_name in ["train", "test", "validation"] :
            encoded_dataset[split_name] = {}
            dataloader = DataLoader(self.ds[split_name], 
                batch_size = self.config.preprocess_batch_size, shuffle = True
            )

            for batch in tqdm(dataloader,desc=f"Tokenize {split_name}"): 
                encoding_output = self.__encoding_data_function(batch)
                for key in encoding_output:
                    if key not in encoded_dataset[split_name].keys():
                        encoded_dataset[split_name][key] = []
                    encoded_dataset[split_name][key].extend(
                        encoding_output[key]
                    )
        end = time()
        self.encoded_dataset = DatasetDict({
            "train" : Dataset.from_dict(encoded_dataset["train"]),
            "test" : Dataset.from_dict(encoded_dataset["test"]),
            "validation" : Dataset.from_dict(encoded_dataset["validation"])
        })
        self.logger.info(f">>> Tokenization - Done ({end - start:.2f})")

    def load_model(self) -> None :
        start = time()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            problem_type = "multi_label_classification", 
            num_labels = self.n_labels,
            id2label = self.id2label, 
            label2id = self.label2id
        ).to(self.config.device)
        end = time()

        if self.config.only_train_classifier : 
            self.logger.info(("--WARNING--\n"
                "You are only training the classifier, the "
                "embeddingmodel is frozen"
            ))
            for name, param in self.model.named_parameters():
                if name.startswith("classifier") : param.requires_grad = True
                else : param.requires_grad = False
        self.logger.info(f">>> Model loading - Done ({end - start :.2f})")
    
    def __subsetting_ds(self) -> None:
        self.logger.info((
            "--WARNING--\n"
            "for dev purposes you are only using a subset of the "
            "dataset you loaded"
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
        self.logger.info(f">>> Training start {datetime.today().strftime('%d-%m-%Y; %H:%M')}")
        start = time()
        try : 
            trainer.train()
            self.trainer_log_history = trainer.state.log_history
        except :
            self.logger.info((
                "--ERROR--\n"
                ">>> Training stopped abruptely "
                f"{datetime.today().strftime('%d-%m-%Y; %H:%M')}"
            ))
        finally:
            end = time()
            self.logger.info(f">>> Training - Done ({end - start:.2f})")

            del trainer
            gc.collect()
            if self.config.device == "cuda": 
                synchronize();empty_cache();ipc_collect()

        try : 
            # TODO dig deeper
            log_filename = self.config.training_args.to_dict()["output_dir"] + "/" +\
                        self.config.training_args.to_dict()["logging_dir"] + ".csv"
            DataFrame(self.trainer_log_history).to_csv(log_filename)
            self.logger.info(f">>> Saving Training Logs - Done")
        except : 
            self.logger.error("Couldn't save the logs")

    def test_f1(self) -> None:
        test_dataset = Dataset.from_dict({
            'input_ids' : self.encoded_dataset["test"]["input_ids"],
            'attention_mask' : self.encoded_dataset["test"]["attention_mask"],
            'labels' : self.encoded_dataset["test"]["labels"]
        }).with_format("torch")
        
        test_dataloader = DataLoader(test_dataset, batch_size = self.config.batch_size)
        f1 = 0
        self.logger.info(">>> Evaluating the F1 test - Start")
        start = time()
        try:
            for batch in test_dataloader:
                output = self.model(**{
                    'input_ids' : batch["input_ids"].to(device = self.model.device),
                    'attention_mask' : batch["attention_mask"].to(device = self.model.device)
                })
                metrics = multi_label_metrics(
                    output.logits.to(device = "cpu"),
                    batch["labels"].to(device = "cpu")
                )
                f1 += metrics["f1"]
            self.logger.info(">>> Evaluating the F1 test - Done")
        except :
            self.logger.info("### ERROR ### evaluating the test dataset failed")
        finally:
            end = time()
            del test_dataset, test_dataloader, batch
            gc.collect()
            if self.config.device == "cuda" : 
                synchronize();empty_cache();ipc_collect()

            self.logger.info((
                f"--RESULTS TEST F1 ({end - start :.2})--"
                f"{f1 / len(test_dataloader)}"
            ))

    def clean(self):
        del (
            self.model,
            self.ds,
            self.tokenizer
        )
        gc.collect()

    def run(self):
        try :
            self.open_file()
            self.preprocess_data()
            self.split_ds()
            self.tokenize_data()
            if self.config.dev_mode : self.__subsetting_ds()
            self.load_model()
            self.train()
            self.test_f1()
        except : 
            self.logger.info("### ERROR ### something messed up")
        finally :
            self.clean()


