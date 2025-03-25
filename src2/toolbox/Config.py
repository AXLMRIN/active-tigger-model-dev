# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from torch.cuda import is_available as cuda_available
# Native

# Custom

# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Config:
    def __init__(self):
        self.seed : int = 2306406
        self.device = "cuda" if cuda_available() else "cpu"
        # Dataset parameters
        self.dataset_filename : str = "data/316_ideological_book_corpus/ibc.csv" # UPGRADE make it editable 
        self.dataset_label_col : str = "leaning"
        self.dataset_text_col : str = "sentence"
        self.dataset_split_parameters : dict = {
            "proportion_train" : 0.7,
            "proportion_test" : 0.15, 
            "proportion_valid" : 0.15,
            "shuffle" : True, 
            "seed" : self.seed
        }
        self.dataset_n_labels = None
        #Custom classifier settings
        # TODO Implement dynamic
        self.classifier_hiddenlayer_dim = 300
        self.classifier_threshold = 0.5
        # Tokenizer 
        self.tokennizer_settings : dict = {
            "padding" : "max_length",
            "truncation" : True,
            "max_length" : 20,
            "return_tensors" : "pt"
        }
        # Embedding model
        # TODO Implement dynamic
        self.embeddingmodel_name = "google-bert/bert-base-uncased"
        self.embeddingmodel_dim = 768
        self.embeddingmodel_output = "last_hidden_state"
        
    def force_cpu(self) -> None: self.device = "cpu"
    def force_gpu(self) -> None: self.device = "cuda"

    def __str__(self):
        return (
            "Config object"
        )