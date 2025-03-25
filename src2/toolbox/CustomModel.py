# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from torch import Tensor
from torch.cuda import synchronize, ipc_collect, empty_cache
# Native
import gc
# Custom
from .Config import Config
from .CustomEmbedder import CustomEmbedder
from .CustomClassifier import CustomClassifier
from .History import History
from .general import Evaluator
# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class CustomModel:
    def __init__(self, config : Config, embedder : CustomEmbedder, 
                 classifier : CustomClassifier) -> None:
        self.config = config
        self.embedder = embedder
        self.classifier = classifier
        self.history = History()
        self.evaluator = Evaluator(
            self.config.dataset_n_labels,
            self.config.classifier_threshold
        )
    
    def predict(self, entries : list[str]):
        embeddings : Tensor = self.embedder(entries) # shape(batch x config.embeddingmodel_dim)
        logits = self.classifier(embeddings)
        return logits

    def __str__(self) -> str:
        return (
            "Custom Model object"
        )

    def clean(self):
        self.embedder.clean()
        del self.embedder, self.classifier
        gc.collect()
        if self.config.device == "cuda":
            synchronize()
            empty_cache()
            ipc_collect()