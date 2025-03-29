# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from torch.cuda import synchronize, ipc_collect, empty_cache
# Native
from time import time
import gc
# Custom
from .Config import Config
# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class CustomEmbedder:
    def __init__(self, config : Config):
        self.config = config
        #  Load model
        self.model = AutoModel.from_pretrained(
            self.config.embeddingmodel_name
        ).to(self.config.device)
        self.config.embeddingmodel_dim = self.model.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.embeddingmodel_name
        )
    def __call__(self, entries : list[str]) -> Tensor:
        tokenized = self.tokenizer(entries,**self.config.tokennizer_settings)
        output = self.model(**{
            key : tokenized[key].to(device = self.model.device)
            for key in tokenized
        })
        
        if (self.config.embeddingmodel_output == "pooler_output")&\
           ("pooler_output" in output.keys()) : 
            return output.pooler_output
        else:
            return output.last_hidden_state[:,0,:]
    
    def save_to_disk(self, filename):
        self.model.save_pretrained(filename)

    def load_from_disk(self, filename):
        self.model = AutoModel.from_pretrained(filename)

    def train(self):
        for param in self.model.parameters(): param.requires_grad = True
    
    def eval(self):
        for param in self.model.parameters(): param.requires_grad = False

    def clean(self):
        del self.model, self.tokenizer
        gc.collect()
        if self.config.device == "cuda":
            synchronize()
            empty_cache()
            ipc_collect()