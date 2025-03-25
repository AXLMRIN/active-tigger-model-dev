# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import Dataset
from torch import Tensor
from torch.cuda import synchronize, ipc_collect, empty_cache
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import nll_loss
# Native
import gc
from tqdm import tqdm
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
        self.classifier = classifier.to(device = self.config.device)
        self.history = History()
        self.evaluator = Evaluator(
            self.config.dataset_n_labels,
            self.config.classifier_threshold
        )
        self.optimizer_embedding = SGD(
            self.embedder.model.parameters(),
            lr = self.config.model_train_learning_rate,
            momentum = self.config.model_train_momentum
        )
        self.optimizer_classifier = SGD(
            self.classifier.parameters(),
            lr = self.config.model_train_learning_rate,
            momentum = self.config.model_train_momentum
        )
        self.loss_function = nll_loss
    
    def predict(self, entries : list[str]):
        embeddings : Tensor = self.embedder(entries) # shape(batch x config.embeddingmodel_dim)
        logits = self.classifier(embeddings)
        return logits

    def train_loop(self, loader : DataLoader, epoch : int) -> None: 
        for batch in tqdm(loader, 
                          desc = "Training loop", leave = False, position = 1):
            self.optimizer_classifier.zero_grad()
            self.optimizer_embedding.zero_grad()

            prediction_logits = self.predict(batch["text"])
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"])
            self.history.append_loss_train(epoch, loss.item())
            loss.backward()

            self.optimizer_classifier.step()
            self.optimizer_embedding.step()

    def validation_loop(self, loader : DataLoader, epoch : int) -> None:
        loss_value : float = 0
        for batch in tqdm(loader, 
                          desc = "Testing loop", leave = False, positin = 1):
            prediction_logits = self.predict(batch["text"])
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"],reduction = "sum")
            loss_value += loss.item()
        loss_value = loss_value / len(loader.dataset)
        self.history.append_loss_validation(epoch, loss_value = loss_value)

    
    def train(self, train_dataset : Dataset, validation_dataset : Dataset) -> None:
        train_loader = DataLoader(
            train_dataset, 
            shuffle = True, 
            batch_size = self.config.model_train_batchsize
        )
        validation_loader = DataLoader(
            validation_dataset, 
            shuffle = True, 
            batch_size = self.config.model_train_batchsize
        )
        for epoch in tqdm(range(self.config.model_train_n_epoch),
                          desc = "Train Epoch", leave = True, position = 0):
            self.train_loop(train_loader, epoch)
            self.validation_loop(validation_loader, epoch)

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