# IMPORTS ######################################################################
import json
import pandas as pd
from ..general import pretty_printing_dictionnary, shuffle_list
from ..CustomLogger import CustomLogger
from datasets import Dataset, DatasetDict, load_from_disk
from collections.abc import Callable
from pandas.api.typing import DataFrameGroupBy
from typing import Any
import os
# SCRIPTS ######################################################################
class DataHandler : 
    """
    """

    def __init__(self, filename : str, text_column : str, label_column : str, 
        logger : CustomLogger) -> None : 
        """
        """
        self.__filename : str = filename
        self.__text_column : str = text_column
        self.__label_column : str = label_column
        self.__logger = logger

        # Variables that will be define later
        (self.__df, self.len, self.columns, self.id2label, self.label2id, 
        self.n_labels, self.n_entries_per_label, self.N_train, self.N_eval, 
        self.N_test, self.__ds) = (None, ) * 11
        
        # Status
        self.status : dict[str:bool] = {
            'open' : False,
            'preprocess' : False,
            'split' : False,
            "encoded" : False
        }
    
    def __str__(self) -> str:
        return_string : str = (
            f"------------\n"
            f"DataHandler, {self.__filename}\n"
            f"------------\n"
            f"Status : {pretty_printing_dictionnary(self.status)}\n"
        )
        if self.status["open"] : 
            return_string += (
                "\n"
                f"DF : ({self.len} x {self.columns})\n"
                f"{self.n_labels} labels ({list(self.label2id.keys())})\n"
                f"Number of element per label : \n"
                f"{pretty_printing_dictionnary(self.n_entries_per_label)}"
            )

        return return_string


    def open_data(self, extra_columns_to_keep : list[str] = []) -> None:
        """Open the file, replace the columns for lighter pipeline. Fetch the labels
        and initialise the label2id, id2label, n_labels and n_entries_per_label
        variables.
        No error catching yet
        """
        # UPGRADE : deal with errors
        # Open and adapt for the pipeline
        replace_columns : dict[str, str] = {
            self.__text_column : "TEXT",
            self.__label_column : "LABEL"
        }
        self.__df : pd.DataFrame = pd.read_csv(f"{self.__filename}").\
            rename(replace_columns, axis = 1).\
            loc[:, ["TEXT", "LABEL", *extra_columns_to_keep]].\
            dropna().\
            sample(frac = 1) # Shuffle

        self.len : int = len(self.__df)
        self.columns : list[str] = list(self.__df.columns)

        # Fetch labels
        self.label2id : dict[str:int]= {}
        self.id2label : dict[int:str]= {}
        self.n_entries_per_label : dict[str:int] = {}

        for id, (label, sub_df) in enumerate(self.__df.groupby("LABEL")):
            self.label2id[label] = id
            self.id2label[id] = label
            self.n_entries_per_label[label] = len(sub_df)

        self.n_labels : int = len(self.label2id)
            
        self.status["open"] = True
        
        # Logging
        self.__logger((f"[DataHandler] Data openning - Done \n"
            f"File : {self.__filename}\n"
            f"{self.n_labels} labels ({list(self.label2id.keys())})\n"
            f"{self.n_entries_per_label}"))
    
    def preprocess(self, function : Callable[[str], str] | None = None) -> None: 
        """
        """
        if function is None : 
            pass
        else : 
            self.__df["TEXT"] = self.__df["TEXT"].apply(function)

            # Logging
            self.__logger("[DataHandler] Data preprocessing - Done")
        
        self.status["preprocess"] = True
    
    def split(self, ratio_train : float = 0.7, 
              ratio_eval : float = 0.15,
              stratify_columns : str | None = None, 
              ) -> None: 
        """
        """
        ##
        if not(stratify_columns is None) : 
            strata : DataFrameGroupBy = self.__df.groupby(stratify_columns)
            
            max_elements_per_stratum : int = strata.size().min()
            df_to_select_from : pd.DataFrame = strata.sample(
                n = max_elements_per_stratum)
            n_entries_available : int = len(df_to_select_from)

        else : 
            df_to_select_from : pd.DataFrame = self.__df
            n_entries_available : int = self.len
        
        # Calculate the number of elements in the train, eval and test set
        # With regard to the number of entries available (depends on the stratification)
        self.N_train : int = int(ratio_train * n_entries_available)
        self.N_eval : int  = int(ratio_eval  * n_entries_available)
        self.N_test : int  = n_entries_available - self.N_train - self.N_eval
        
        shuffled_index = shuffle_list(df_to_select_from.index.to_list())
        index_train = shuffled_index[:self.N_train]
        index_eval  = shuffled_index[self.N_train:-self.N_test]
        index_test  = shuffled_index[-self.N_test:]

        df_train : pd.DataFrame = df_to_select_from.loc[index_train, :]
        df_eval : pd.DataFrame  = df_to_select_from.loc[index_eval, :]
        df_test : pd.DataFrame  = df_to_select_from.loc[index_test, :]

        self.__ds = DatasetDict({
            "train" : Dataset.from_pandas(df_train),
            "eval"  : Dataset.from_pandas(df_eval),
            "test"  : Dataset.from_pandas(df_test),
        })

        self.status["split"] = True

        # Logging
        self.__logger((f"[DataHandler] Data encoding - Done\n"
            f"Split dataset (with stratification {stratify_columns})\n"
            f"N_train : {self.N_train}; N_eval : {self.N_eval}; "
            f"N_test : {self.N_test}"))
    
    def encode(self, tokenizer, tokenizing_parameters : dict) :
        """
        """
        
        for ds_name in ["train","eval", "test"] : 
            input_ids_list : list[list[int]] = []
            attention_mask_list : list[list[bool]] = []
            labels_list : list[list[bool]] = []
            for batch_of_rows in self.__ds[ds_name].batch(64,drop_last_batch=False) :
                # row : {'text' : list[str], 'label' : list[str]} 
                tokens = tokenizer(
                    batch_of_rows["TEXT"], **tokenizing_parameters)
                
                input_ids_list.extend(tokens.input_ids)
                attention_mask_list.extend(tokens.attention_mask)
                labels_list.extend(self.__make_labels_matrix(batch_of_rows["LABEL"]))

            self.__ds[ds_name] = self.__ds[ds_name].add_column("input_ids", input_ids_list)
            self.__ds[ds_name] = self.__ds[ds_name].add_column("attention_mask", attention_mask_list)
            self.__ds[ds_name] = self.__ds[ds_name].add_column("labels", labels_list)
        
        self.status["encoded"] = True
        
        # Logging
        self.__logger("[DataHandler] Data encoding - Done")
    
    def debug_mode(self):
        """
        """
        self.__ds["train"] = \
            self.__ds["train"].select(range(20))
        self.__ds["eval"] = \
            self.__ds["eval"].select(range(20))
        self.__ds["test"] = \
            self.__ds["test"].select(range(20))
        
        # Logging
        self.__logger(("[DataHandler] DEBUG MODE, only 20 elements per split "
                       "(train, eval, test)"))

    def __make_labels_matrix(self, labels) -> list[list[float]]:
        """
        """
        return [
            [float(id == self.label2id[label]) for id in range(self.n_labels)]
            for label in labels
        ]
    
    def get_encoded_dataset(self, ds_name : str) -> Dataset :
        """
        """
        return self.__ds[ds_name] 

    def save_all(self, foldername : str) -> None:
        """
        """
        # Save the config
        if os.path.exists(f"{foldername}/data") : 
            os.remove(f"{foldername}/data")
        os.mkdir(f"{foldername}/data")
        with open(f"{foldername}/data/DataHandler_config.json", "w") as file:
            config = {
                "date" : pd.Timestamp.today().strftime("%Y-%m-%d"), # FIXME '2025-31-06/14/25'
                "label2id" : self.label2id, 
                "status" : self.status, 
                "filename" : self.__filename,
                "text_column" : self.__text_column,
                "label_column" : self.__label_column, 
                "len" : self.len,
                "columns" : self.columns
            }
            json.dump(config, file, ensure_ascii=True, indent=4)

        # Save the dataset
        self.__ds.save_to_disk(f"{foldername}/data")

        # Logging
        self.__logger("[DataHandler] Data configuration saved")

    def routine(self, 
        preprocess_function : Callable[[str], str]|None = None,
        ratio_train : float = 0.7, ratio_eval : float = 0.15,
        stratify_columns : list[str]|None = None
        ) -> None: 
        """
        """
        self.__logger("[DataHandler] Routine start ---", skip_line = "before")
        try : 
            self.open_data()
        except Exception as e:
            raise ValueError(f"Data could not be open.\n\nError:\n{e}")
        ###
        try : 
            self.preprocess(preprocess_function)
        except Exception as e:
            raise ValueError(f"Data could not be preprocessed.\n\nError:\n{e}")
        ###f
        try:
            self.split(ratio_train, ratio_eval, stratify_columns)
        except Exception as e:
            raise ValueError(f"Data could not be split.\n\nError:\n{e}")
        
        self.__logger("[DataHandler] Routine finish ---", skip_line = "after")
    
    def debug(self):#TODELETE
        return self.__ds