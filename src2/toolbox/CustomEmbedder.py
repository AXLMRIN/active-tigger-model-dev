# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
# Native
from time import time
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.embeddingmodel_name
        )
    def __call__(self, entries : list[str]) -> Tensor:
        tokenized = self.tokenizer(entries,**self.config.tokennizer_settings)
        output = self.model(**{
            key : tokenized[key].to(device = self.model.device)
            for key in tokenized
        })
        print(output.keys())
        if (self.config.embeddingmodel_output == "pooler_output")&\
           ("pooler_output" in output.keys()) : 
            return output.pooler_output
        else:
            return output.last_hidden_state[:,0,:]
        