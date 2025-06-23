# IMPORTS ######################################################################
from .DataHandler import DataHandler
from torch.cuda import is_available as cuda_available
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from transformers.trainer_utils import TrainOutput
from .functions import compute_metrics
import os
from ..general import pretty_number, pretty_printing_dictionnary, clean
from time import time
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################

class CustomTransformersPipeline:
    """
    """
    def __init__(self, 
            data : DataHandler, 
            model_name : str, 
            logger : CustomLogger,
            device : str|None = None, 
            tokenizer_max_length : int = 128,
            total_batch_size : int = 64, 
            batch_size_device : int = 8, 
            num_train_epochs : int = 3, 
            learning_rate : float = 2e-5,
            weight_decay : float = 0.01, 
            warmup_ratio : float = 0.1, 
            optimizer : str = "adamw_torch", 
            output_dir : str|None = None,
            load_best_model_at_end : bool = True,
            disable_tqdm : bool = False
        ) -> None:
        """
        """
        self.__data : DataHandler = data
        self.model_name : str = model_name
        self.__logger : CustomLogger = logger
        if device is None : 
            self.device : str = "cuda" if cuda_available() else "cpu"
        else : 
            self.device : str = device

        (self.model, self.tokenizer) = (None, )*2

        # Parameters 
        self.tokenizing_parameters : dict = {
            'padding' : 'max_length',
            'truncation' : True,
            'max_length' : tokenizer_max_length
        }

        if output_dir is None : 
            self.output_dir = (f"./models/{self.model_name}/") # FIXME could be done in a cleaner way
            if os.path.isdir(self.output_dir) : 
                n_elements_in_output_dirs : int = len(
                    os.listdir(self.output_dir)
                )
            else : 
                n_elements_in_output_dirs : int = 0
            self.output_dir  += f"{pretty_number(n_elements_in_output_dirs + 1)}"
        else : 
            self.output_dir = output_dir

        self.training_args = TrainingArguments(
            # bf16=True, # NOTE investigate
            # Hyperparameters
            num_train_epochs = num_train_epochs,
            learning_rate = learning_rate,
            weight_decay  = weight_decay,
            warmup_ratio  = warmup_ratio,
            optim = optimizer,
            # Second order hyperparameters
            per_device_train_batch_size = batch_size_device,
            per_device_eval_batch_size = batch_size_device,
            gradient_accumulation_steps = \
                int(total_batch_size/ batch_size_device),
            # Metrics
            metric_for_best_model="f1_macro",
            # Pipe
            output_dir = self.output_dir,
            overwrite_output_dir=True,
            eval_strategy = "epoch",
            logging_strategy = "epoch",
            save_strategy = "epoch",
            torch_empty_cache_steps = int(
                self.__data.n_labels * self.__data.N_train / batch_size_device),
            load_best_model_at_end = load_best_model_at_end,
            save_total_limit = num_train_epochs + 1,

            disable_tqdm = disable_tqdm
        )

        self.status : dict[str:bool] = {
            "loaded" : False,
            "trained" : False
        }

    def __str__(self) -> str :
        out = (
            f"---------------------------\n"
            f"CustomTransformersPipeline, {self.model_name}\n"
            f"---------------------------\n"
            f"{pretty_printing_dictionnary(self.status)}\n"
            f"Main parameters :\n"
            f"\t- Learning rate : {self.training_args.learning_rate}\n"
            f"\t- Weight Decay : {self.training_args.weight_decay}\n"
            f"\t- Warmup Ratio : {self.training_args.warmup_ratio}\n"
            f"\t- Optimizer : {self.training_args.optim}\n"
            f"\t- Output Directory : {self.training_args.output_dir}\n"
        )
        return out

    def __save_model_name(self) -> None : 
        """
        """
        with open(f"{self.training_args.output_dir}/model_name.txt", "w") as file:
            file.write(self.model_name)

    def load_tokenizer_and_model(self, skip_model : bool = False) -> None:
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)   

        if not skip_model: # TODELETE
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                problem_type = "multi_label_classification",
                num_labels   = self.__data.n_labels,
                id2label     = self.__data.id2label,
                label2id     = self.__data.label2id).\
                to(device = self.device)
            
            # update the max_length 
            self.tokenizing_parameters["max_length"] = min(
                self.tokenizing_parameters["max_length"],
                self.model.config.max_position_embeddings
            )
        
        self.status["loaded"] = True

        # Logging
        self.__logger((f"[CustomTransformersPipeline] Model and Tokenizer loading"
            " - Done"))
    def train(self) -> TrainOutput:
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.__data.get_encoded_dataset("train"),
            eval_dataset = self.__data.get_encoded_dataset("eval"),
            compute_metrics = compute_metrics
        )
        t1 = time()
        output = trainer.train()
        # TODO Find a way to do something about it
        t2 = time() 
        
        # Logging
        self.__logger((f"[CustomTransformersPipeline] Model Training - Done "
            f"({t2 - t1:.0f} seconds)"))
        return output
    
    def routine(self, debug_mode : bool = False):
        """
        """
        self.__logger("[CustomTransformersPipeline] Routine start ---", 
            skip_line="before")
        self.__logger((f"[CustomTransformersPipeline] Pipeline creation - Done\n"
            f"Model {self.model_name} on {self.device}\n"
            f"Output directory : {self.output_dir}"))

        try : 
            self.load_tokenizer_and_model()
        except Exception as e:
            del self.model, self.__data, self.tokenizer
            clean()
            raise ValueError((f"Training Pipeline could not load the tokenizer "
                              f"and model.\n\nError:\n{e}"))
        ###
        try :
            self.__data.encode(self.tokenizer, self.tokenizing_parameters)
        except Exception as e:
            del self.model, self.__data, self.tokenizer
            clean()
            raise ValueError((f"Training Pipeline could not encode the dataset."
                              f"\n\nError:\n{e}"))
        ###
        if debug_mode : 
            try : 
                self.__data.debug_mode()
            except Exception as e:
                del self.model, self.__data, self.tokenizer
                clean()
                raise ValueError((f"Training Pipeline could not set data into"
                                  f" debugging mode.\n\nError:\n{e}"))
        ###
        try:
            output = self.train()
        except Exception as e:
            del self.model, self.__data, self.tokenizer
            clean()
            raise ValueError((f"Training Pipeline could not train the model."
                              f"\n\nError:\n{e}"))
        ###
        try:
            self.__data.save_all(self.training_args.output_dir)
        except Exception as e:
            del self.model, self.__data, self.tokenizer
            clean()
            raise ValueError((f"Training Pipeline (Data) could not save data."
                              f"\n\nError:\n{e}"))
        ###
        try: 
            self.__save_model_name()
        except Exception as e:
            del self.model, self.__data, self.tokenizer
            clean()
            raise ValueError((f"Training Pipeline could not save the model name."
                              f"\n\nError:\n{e}"))
        ###
        del self.model, self.__data, self.tokenizer
        clean()
        
        self.__logger("[CustomTransformersPipeline] Routine finish ---", 
            skip_line="after")
        return output