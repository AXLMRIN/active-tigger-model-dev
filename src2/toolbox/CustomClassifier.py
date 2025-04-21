# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from torch import nn, Tensor
import torch.nn.functional as F
# Native

# Custom
from .Config import Config
# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class CustomClassifier(nn.Module):
    def __init__(self, config : Config):
        # NOTE removed the several layers to only keep one layer
        super(CustomClassifier, self).__init__()
        
        self.config = config
        if self.config.classifier_mode == "single_layer" : 
            self.f_single = nn.Linear(
                self.config.embeddingmodel_dim,
                self.config.dataset_n_labels
            )
        elif self.config.classifier_mode == "multi_layer" : 
            self.fc1 = nn.Linear(
                self.config.embeddingmodel_dim, 
                self.config.classifier_hiddenlayer_dim
            )
            self.fc2 = nn.Linear(
                self.config.classifier_hiddenlayer_dim,
                self.config.dataset_n_labels
            )

    def forward(self, x : Tensor):
        if self.config == "single_layer" : 
            x = x.view(-1, self.config.embeddingmodel_dim)
            output = F.log_softmax(
                self.f_single(x),
                dim = 0
            )
        elif self.config == "multi_layer" : 
            x = x.view(-1, self.config.embeddingmodel_dim)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training = self.training)
            x = x.view(-1, self.config.classifier_hiddenlayer_dim)
            output = F.log_softmax(
                self.fc2(x),
                dim = 0
            )
        return output
