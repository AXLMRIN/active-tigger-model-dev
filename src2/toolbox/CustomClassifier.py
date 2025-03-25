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
        super(CustomClassifier, self).__init__()
        
        self.config = config
        self.fc1 = nn.Linear(
            self.config.embeddingmodel_dim, 
            self.config.classifier_hiddenlayer_dim
        )
        self.fc2 = nn.Linear(
            self.config.classifier_hiddenlayer_dim,
            self.config.dataset_n_labels
        )

    def forward(self, x : Tensor):
        x = x.view(-1, self.config.embeddingmodel_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = x.view(-1, self.config.classifier_hiddenlayer_dim)
        output = F.log_softmax(
            self.fc2(x),
            dim = 0
        )
        return output
