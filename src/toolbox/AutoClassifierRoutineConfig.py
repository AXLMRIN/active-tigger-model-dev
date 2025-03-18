# TODO write a banner
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from torch.cuda import is_available as gpu_available
from transformers import TrainingArguments
from mergedeep import merge

# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class AutoClassifierRoutineConfig: 
    def __init__(self, 
        files : dict[str:str],
        sentence_col : str, 
        label_col : str, 
        batch_size : int = 32, 
        num_train_epochs : int = 10,
        only_train_classifier : bool = False, 
        dev_mode : bool = False,
        **kwargs) -> None:
        """
        Config for AutoClassifierRoutine
        """
        # Routine functions related : 
        self.dev_mode = dev_mode
        self.only_train_classifier = only_train_classifier
        # Dataset related
        if "open_s3" in files: 
            self.filename_open_s3 : str = files["open_s3"]
        else: self.filename_open_s3 = None
        if "open_local" in files: 
            self.filename_open_local : str = files["open_local"]
        else : self.filename_open_local = None

        self.sentence_col : str = sentence_col
        self.label_col : str = label_col
        # Batch related
        self.batch_size : int = batch_size
        # Model Related
        self.model_name = "answerdotai/ModernBERT-base"
        self.device : str = "cuda" if gpu_available() else "cpu"
        self.att_implementation : str = "sdpa" # FIXME Not implemented for now TODO look deeper
        
        training_args : dict = {
            # Parameters that are customisable
            "output_dir" : files["output_dir"],
            "logging_dir" : files["output_dir"] + "_log", 
            "save_strategy" : "no",
            "num_train_epochs" : 1 if self.dev_mode else num_train_epochs,
            # Fixed parameters
            "overwrite_output_dir" : True,
            "eval_strategy" : "epoch",
            "learning_rate" : 2e-5,
            "per_device_train_batch_size" : self.batch_size,
            "per_device_eval_batch_size" : self.batch_size,
            "weight_decay" : 0.01,
            "logging_first_step":False,
            "logging_strategy":"epoch",
            "save_safetensors" : False
        }
        if "training_args" in kwargs:
            training_args = merge(training_args, kwargs["training_args"])
        self.training_args = TrainingArguments(**training_args) # TODO continue digging the parameters
        # Tokenizer related
        self.tokenizer_settings : dict = {
            "padding" : "max_length",
            "truncation" : True,
            "max_length" : 128 if "tokenizer_max_length" not in kwargs else kwargs["tokenizer_max_length"]
        }