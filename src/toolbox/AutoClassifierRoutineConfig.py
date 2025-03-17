# TODO write a banner
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from torch.cuda import is_available as gpu_available
from transformers import TrainingArguments

# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class AutoClassifierRoutineConfig: 
    def __init__(self, 
        files : dict[str:str],
        sentence_col : str, 
        label_col : str, 
        batch_size : int = 32, 
        only_train_classifier : bool = False, 
        dev_mode : bool = False ) -> None:
        # Routine functions related : 
        self.dev_mode = dev_mode
        self.only_train_classifier = only_train_classifier
        # Dataset related
        self.filename_open_s3 : str = "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data/ibc.csv", 
        self.filename_open_local : str = "data/316_ideological_book_corpus/ibc.csv"
        self.sentence_col : str = sentence_col
        self.label_col : str = label_col
        # Model Related
        self.model_name = "answerdotai/ModernBERT-base"
        self.device : str = "cuda" if gpu_available() else "cpu"
        self.att_implementation : str = "sdpa" # FIXME Not implemented for now TODO look deeper
        self.training_args = TrainingArguments(
            # Parameters that are customisable
            output_dir = files["output_dir"],
            save_strategy = "no" if self.dev_mode else "epoch",
            num_train_epochs = 1 if self.dev_mode else 10,
            # Fixed parameters
            overwrite_output_dir=True,
            eval_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
        ) # TODO continue digging the parameters
        # Batch related
        self.batch_size : int = batch_size
        # Tokenizer related
        self.tokenizer_settings : dict = {
            "padding" : "max_length",
            "truncation" : True,
            "max_length" : 128
        }