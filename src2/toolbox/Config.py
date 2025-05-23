# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from torch.cuda import is_available as cuda_available
# Native
import json 

# Custom

# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Config(object):
    def __init__(self):
        self.seed : int = 2306406
        self.device = "cuda" if cuda_available() else "cpu"
        # Dataset parameters
        self.dataset_filename : str = "dataUNSAFE/ibc.csv" # UPGRADE make it editable 
        self.dataset_label_col : str = "leaning"
        self.dataset_text_col : str = "sentence"
        self.dataset_split_parameters_DEPRECATED : dict = {
            "proportion_train" : 0.7,
            "proportion_test" : 0.15, 
            "proportion_valid" : 0.15,
            "shuffle" : True, 
            "seed" : self.seed
        }
        self.dataset_split_parameters : dict = {
            "N_trainP" : 0.7,
            "N_testP" : 0.15, 
            "N_evalP" : 0.15,
            "seed" : self.seed
        }
        self.dataset_n_labels = None
        self.dataset_label2id = None
        self.dataset_id2label = None
        #Custom classifier settings
        self.classifier_mode = "single_layer" # "single_layer" or "multi_layer"
        self.classifier_hiddenlayer_dim = 50 # Not used if classifier_mode = "single_layer"
        self.classifier_threshold = 0.3 # NOTE Never used
        self.classifier_save_filename = "custom_classifier_save.pt"

        # Tokenizer 
        self.tokennizer_settings : dict = {
            "padding" : "max_length",
            "truncation" : True,
            "max_length" : None, # NOTE this was modified to urge the user to choose the max_length wisely
            "return_tensors" : "pt"
        }

        # Embedding model
        self.embeddingmodel_name = "google-bert/bert-base-uncased"
        self.embeddingmodel_dim = None
        self.embeddingmodel_output = "last_hidden_state"
        self.embeddingmodel_save_filename = f"bert-base-uncased_custom_save"
        
        # Model
        self.model_save_best_model = True
        self.model_loss_parameters = {
            "reduction" : "sum",
            "weight" : None
        }
        self.model_train_batchsize = 32
        self.model_train_n_epoch = 5
        self.model_train_embedding_optimizer = "Adam"
        self.model_train_embedding_adam_parameters = {
            "lr" : 1e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 0                          # Default
        }
        self.model_train_embedding_adamw_parameters = {
            "lr" : 1e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 1e-2                       # Default
        }
        self.model_train_embedding_adamax_parameters = {
            "lr" : 2e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 0                          # Default
        }

        self.model_train_classifier_optimizer = "SGD"
        self.model_train_classifier_sgd_parameters = {
            "lr" : 1e-3,                                # Default
            "momentum" : 0,                             # Default
            "dampening" : 0,                            # Default
            "weight_decay" : 0                          # Default
        }
        self.model_train_classifier_adam_parameters = {
            "lr" : 1e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 0                          # Default
        }
        self.model_train_classifier_adamw_parameters = {
            "lr" : 1e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 1e-2                       # Default
        }
        self.model_train_classifier_adamax_parameters = {
            "lr" : 2e-3,                                # Default
            "betas" : (0.9,0.999),                      # Default
            "eps" : 1e-8,                               # Default
            "weight_decay" : 0                          # Default
        }

        #History
        self.history_foldersave : str = "./HISTORY_SAVE"
        
    def force_cpu(self) -> None: self.device = "cpu"
    def force_gpu(self) -> None: self.device = "cuda"

    def __getattr__(self, item):return super(Config, self).__setattr__(item, 'orphan')

    def save(self) : 
        with open(self.history_foldersave + "/config.json", "w") as file:
            json.dump(self.__dict__, file, sort_keys = True, indent = 4)
            
    def __str__(self):
        return (
            "Config object"
        )