# IMPORTS ######################################################################
import pandas as pd
from ..general import pretty_printing_dictionnary, shuffle_list
from datasets import Dataset, DatasetDict
from collections.abc import Callable
from pandas.api.typing import DataFrameGroupBy
from typing import Any
from .. import ROOT_DATA
from transformers.tokenization_utils_base import BatchEncoding
# SCRIPTS ######################################################################
class DataHandler : 
    """
    """

    def __init__(self, filename : str, text_column : str, label_column : str) : 
        """
        """
        self.__filename : str = filename
        self.__text_column : str = text_column
        self.__label_column : str = label_column

        # Variables that will be define later
        (self.__df, self.len, self.columns, self.id2label, self.label2id, 
        self.n_labels, self.n_entries_per_label, self.N_train, self.N_eval, 
        self.N_test, self.__ds, self.__ds_encoded) = (None, ) * 12
        
        # Status
        self.status : dict[str:bool] = {
            'open' : False,
            'preprocess' : False,
            'split' : False,
            "encoded" : False
        }
    
    def __str__(self) -> str:
        return_string : str = (
            f"DataHandler, {self.__filename}\n"
            f"Status : {pretty_printing_dictionnary(self.status)}\n"
        )
        if self.status["open"] : 
            return_string += (
                "\n"
                f"Columns : {self.columns}\n"
                f"Size : {self.len}\n"
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
        self.__df : pd.DataFrame = pd.read_csv(f"{ROOT_DATA}/{self.__filename}").\
            rename(replace_columns, axis = 1).\
            loc[:, ["TEXT", "LABEL", *extra_columns_to_keep]].\
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
    
    def preprocess(self, function : Callable[[str], str] | None = None) -> None: 
        """
        """
        if function is None : 
            pass
        else : 
            self.__df["TEXT"] = self.__df["TEXT"].apply(function)
        
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
        # With regard to the subset of ???
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
    
    def encode(self, tokenizer, tokenizing_parameters : dict) :
        """
        """
        self.__ds_encoded = DatasetDict()
        for ds_name in ["train","eval", "test"] : 
            ds_encoded_ds_name : dict[list[str|int]] = {
                "INPUT_IDS" : [],
                "ATTENTION_MASK" : [],
                "LABELS" : []
            }
            for batch_of_rows in self.__ds[ds_name].batch(64) :
                # row : {'text' : list[str], 'label' : list[str]} 
                tokens : BatchEncoding = tokenizer(
                    batch_of_rows["TEXT"], **tokenizing_parameters)
                
                ds_encoded_ds_name["INPUT_IDS"].extend(tokens.input_ids)
                ds_encoded_ds_name["ATTENTION_MASK"].extend(tokens.attention_mask)

                ds_encoded_ds_name["LABELS"].extend(
                    self.make_labels_matrix(batch_of_rows["LABEL"]))

            self.__ds_encoded[ds_name] = Dataset.from_dict(ds_encoded_ds_name)
        
        self.status["encoded"] = True
    
    def debug_mode(self):
        if self.__ds_encoded is None: 
            print("the dataset is not yet encoded")
        else : 
            # Debug
            self.__ds_encoded["train"] = \
                self.__ds_encoded["train"].select(range(20))
            self.__ds_encoded["eval"] = \
                self.__ds_encoded["eval"].select(range(20))

    def make_labels_matrix(self, labels) -> list[list[float]]:
        """
        """
        return [
            [float(id == self.label2id[label]) for id in range(self.n_labels)]
            for label in labels
        ]

    def routine(self, 
        preprocess_function : Callable[[str], str]|None = None,
        ratio_train : float = 0.7, ratio_eval : float = 0.15,
        stratify_columns : list[str]|None = None
        ) -> None: 
        """
        """
        self.open_data()
        self.preprocess(preprocess_function)
        self.split(ratio_train, ratio_eval, stratify_columns)

    def debug(self):
        #TODELETE
        return self.__ds