# IMPORTS ######################################################################
from collections.abc import Callable
import json
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
from pandas.api.typing import DataFrameGroupBy

from ..general import pretty_printing_dictionnary, shuffle_list
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class DataHandler : 
    """DataHandler is an object that collects and process data before training 
    models with CustomTransformersPipeline (based on AutoModelForSequenceClassification).

    The main features and functions : 
        - collects the data and retrieve the labels
        - allows preprocessing texts
        - splits data (train, eval, test), with possible stratification
        - saves data
    
    The routine function proceeds to all these steps.
    """

    def __init__(self, filename : str, text_column : str, label_column : str, 
        logger : CustomLogger) -> None : 
        """Builds the DataHandler object. 
        
        Parameters:
        -----------
            - filename (str): full path to the file.
            - text column (str): column for the text data, will be replaced by TEXT.
            - label column (str): column for the labels, will be replaced by LABEL.
            - logger (CustomLogger): will give information as the data is processed.

        Returns:
        --------
            /

        Inisialised variables:
        ----------------------
            DATACOLLECTION
            - self.__filename (str): full path to the file.
            - self.__text column (str): column for the text data, will be replaced by TEXT.
            - self.__label column (str): column for the labels, will be replaced by LABEL.
            - self.__df (pd.DataFrame) : data collected, columns : [TEXT, LABEL, + extra columns].
            - self.len (int): size of the __df.
            - self.columns (list[str]): columns of the __df
            
            MODELTRAINING
            - self.n_labels (int): number of labels
            - self.id2label (dict[int:str]): binder label indices to text labels
            - self.label2id (dict[str:int]): binder text labels to label indices
            - self.n_entries_per_label (dict[str:int]): number of elements per label
            - self.__ds (DatasetDict): data used during training
            - self.N_train (int): number of elements in the train set
            - self.N_eval (int): number of elements in the eval set
            - self.N_test (int): number of elements in the test set
            
            COMMUNICATION AND SECURITY
            - self.__logger (CustomLogger): will give information as the data is processed.
            - self.status (dict[str:bool]): dictionnary stating which function 
                have been used, and what step in the routine had been completed
        """
        self.__filename : str = filename
        self.__text_column : str = text_column
        self.__label_column : str = label_column
        self.__logger = logger

        # Variables that will be define later
        (self.__df, self.len, self.columns, self.id2label, self.label2id, 
        self.n_labels, self.n_entries_per_label, self.__ds, self.N_train, 
        self.N_eval, self.N_test) = (None, ) * 11
        
        # Status
        self.status : dict[str:bool] = {
            'open' : False,
            'preprocess' : False,
            'split' : False,
            "encoded" : False
        }
    
    def __str__(self) -> str:
        """Provides insight on the data collected depending on where you're at in 
        the routine.

        Parameters:
        -----------
            /
        
        Returns:
        --------
            - (str)
        """
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

        Parameters:
        -----------
            - extra_columns_to_keep (list[str], default = []): columns of the 
                dataframe that will be kept for preprocessing or stratifying.
        
        Returns:
        --------
            /
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
        """Preprocess data given a preprocessing function. The function must 
        receive a string variable and return a string. If no function is given, 
        nothing happens.

        Parameters:
        -----------
            - function (Callable[[str], str] or None, default = None)
        
        Returns:
        --------
            /
        """
        if function is None : 
            pass
        else : 
            self.__df["TEXT"] = self.__df["TEXT"].apply(function)

            # Logging
            self.__logger("[DataHandler] Data preprocessing - Done")
        
        self.status["preprocess"] = True
    
    def split(self, 
        ratio_train : float = 0.7, 
        ratio_eval : float = 0.15,
        stratify_columns : list[str] | str | None = None, 
        ) -> None: 
        """Splits data in 3 splits (train, eval, test) for training the model. 
        The number of elements in each split is calculated thanks to 2 ratios 
        (ratio_train and ratio_eval). The splits are created from the available 
        elements. 
        If no stratification column is passed, the available elements are all rows 
        in the dataset (__df). 
        If stratify columns are passed, the rows are split with the groupby method,
        then we evaluate the number of rows in the smaller split (= maximum elements
        that can be picked per stratum). From there, a new dataframe is created 
        by picking  max_elements_per_stratum per stratum.
        
        To create the splits, we evaluate the number of elements in each split
        (N_train, N_eval, N_test), and split the dataset (df_to_select_from).
        N_train and N_eval are calculated from the ratios and the number of 
        available entries, N_test is all the rest.

        Parameters:
        -----------
            - ratio_train (float, default = 0.7): proportion of available elements
                that will be picked for the train set
            - ratio_train (float, default = 0.15): proportion of available elements
                that will be picked for the evaluation set
            - stratify columns (list[str], str or None, default = None): the 
                columns to group __df by. The columns must already exist in __df
        
        Returns:
        --------
            /
        """
        if not(stratify_columns is None) : 
            strata : DataFrameGroupBy = self.__df.groupby(stratify_columns)
            
            max_elements_per_stratum : int = strata.size().min()
            # Recreate a dataframe by picking max_elements_per_stratum elements
            # per stratum.
            df_to_select_from : pd.DataFrame = strata.sample(
                n = max_elements_per_stratum)
            n_entries_available : int = len(df_to_select_from)

        else : 
            df_to_select_from : pd.DataFrame = self.__df
            n_entries_available : int = self.len
        
        # Calculate the number of elements in the train, eval and test set
        # With regard to the number of entries available 
        self.N_train : int = int(ratio_train * n_entries_available)
        self.N_eval : int  = int(ratio_eval  * n_entries_available)
        self.N_test : int  = n_entries_available - self.N_train - self.N_eval
        
        # Create indexes to split the dataframe
        shuffled_index = shuffle_list(df_to_select_from.index.to_list())
        index_train = shuffled_index[:self.N_train]
        index_eval  = shuffled_index[self.N_train:-self.N_test]
        index_test  = shuffled_index[-self.N_test:]

        # Split the dataframe
        df_train : pd.DataFrame = df_to_select_from.loc[index_train, :]
        df_eval : pd.DataFrame  = df_to_select_from.loc[index_eval, :]
        df_test : pd.DataFrame  = df_to_select_from.loc[index_test, :]

        # Create the DatasetDict used in the training loop
        self.__ds = DatasetDict({
            "train" : Dataset.from_pandas(df_train),
            "eval"  : Dataset.from_pandas(df_eval),
            "test"  : Dataset.from_pandas(df_test),
        })

        # Logging
        self.status["split"] = True
        self.__logger((f"[DataHandler] Data encoding - Done\n"
            f"Split dataset (with stratification {stratify_columns})\n"
            f"N_train : {self.N_train}; N_eval : {self.N_eval}; "
            f"N_test : {self.N_test}"))
    
    def encode(self, tokenizer, tokenizing_parameters : dict[str:Any] = {}) :
        """Encode the texts for all splits. The __ds must be created beforehand.

        Parameters:
        -----------
            - tokenizer: tokenizer from transformers.AutoTokenizer.from_pretrained
                method.
            - tokenizing_parameters (dict[str:Any], default = {}): Additional 
                parameters to be passed during the tokenization.

        Returns:
        --------
            /
        """
        for ds_name in ["train","eval", "test"] : 
            input_ids_list : list[list[int]] = []
            attention_mask_list : list[list[bool]] = []
            labels_list : list[list[bool]] = []
            for batch_of_rows in self.__ds[ds_name].batch(64,drop_last_batch=False) :
                # row : {'TEXT' : list[str], 'LABEL' : list[str]} 
                tokens = tokenizer(
                    batch_of_rows["TEXT"], **tokenizing_parameters)
                
                input_ids_list.extend(tokens.input_ids)
                attention_mask_list.extend(tokens.attention_mask)
                labels_list.extend(self.__make_labels_matrix(batch_of_rows["LABEL"]))

            self.__ds[ds_name] = self.__ds[ds_name].\
                add_column("input_ids", input_ids_list)
            self.__ds[ds_name] = self.__ds[ds_name].\
                add_column("attention_mask", attention_mask_list)
            self.__ds[ds_name] = self.__ds[ds_name].\
                add_column("labels", labels_list)
        
        # Logging
        self.status["encoded"] = True
        self.__logger("[DataHandler] Data encoding - Done")
    
    def debug_mode(self):
        """Reduce the number of elements per split to 20 so that the debugging is 
        faster.

        Parameters:
        -----------
            /
        
        Returns:
        --------
            /
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

    def __make_labels_matrix(self, labels : list[str]) -> list[list[float]]:
        """This function is adapting the labels (list[str]) to the expected format 
        of labels during the training. Returns float to avoid raising errors during
        the training.

        Example : 
            IN: 
                [label1, label1, label3, label2, label2, label3]
            OUT: 
                [
                    [1.0, 0.0, 0.0]
                    [1.0, 0.0, 0.0]
                    [0.0, 0.0, 1.0]
                    [0.0, 1.0, 0.0]
                    [0.0, 1.0, 0.0]
                    [0.0, 0.0, 1.0]
                ]

        Parameters:
        -----------
            - labels (list[str]): a list of labels to be turned into a matrix.
        
        Returns:
        --------
            - list[list[float]] -- a matrix of floats.
        """
        return [
            [float(id == self.label2id[label]) for id in range(self.n_labels)]
            for label in labels
        ]
    
    def get_encoded_dataset(self, ds_name : str) -> Dataset :
        """getter function
        
        Parameters:
        -----------
            - ds_name (str): name of the split (train, eval or test).
        
        Returns: 
        --------
            - Dataset: corresponding split.
        """
        return self.__ds[ds_name] 

    def save_all(self, foldername : str) -> None:
        """Save the data used during the training (__ds). Saves a custom config as 
        well.
        """
        # Overwrite data first
        if os.path.exists(f"{foldername}/data") : os.remove(f"{foldername}/data")
        os.mkdir(f"{foldername}/data")

        # Save a custom config.
        with open(f"{foldername}/data/DataHandler_config.json", "w") as file:
            config = {
                "date" : pd.Timestamp.today().strftime("%Y-%m-%d"),
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
        """Routine used to open, preprocess and split data before the training.
        The encoding happens during the training.

        Parameters:
        -----------
            PREPROCESS
            - function (Callable[[str], str] or None, default = None)
            
            SPLIT
            - ratio_train (float, default = 0.7): proportion of available elements
                that will be picked for the train set
            - ratio_train (float, default = 0.15): proportion of available elements
                that will be picked for the evaluation set
            - stratify columns (list[str], str or None, default = None): the 
                columns to group __df by. The columns must already exist in __df
        Returns:
        --------
            /
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