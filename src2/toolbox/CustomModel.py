# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import Dataset
from torch import Tensor, no_grad
from torch.cuda import synchronize, ipc_collect, empty_cache
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
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

        if self.config.model_train_embedding_optimizer == "Adam":
            self.optimizer_embedding = Adam(
                self.embedder.model.parameters(),
                lr = self.config.model_train_embedding_learning_rate,
                weight_decay = self.config.model_train_embedding_weight_decay 
            )
        elif self.config.model_train_embedding_optimizer == "SGD":
            self.optimizer_embedding = SGD(
                self.embedder.model.parameters(),
                lr = self.config.model_train_embedding_learning_rate,
                momentum = self.config.model_train_embedding_momentum,
                weight_decay = self.config.model_train_embedding_weight_decay
            )
            
        if self.config.model_train_classifier_optimizer == "Adam":
            self.optimizer_classifier = Adam(
                self.classifier.parameters(),
                lr = self.config.model_train_classifier_learning_rate,
                weight_decay = self.config.model_train_classifier_weight_decay 
            )
        elif self.config.model_train_classifier_optimizer == "SGD":
            self.optimizer_classifier = SGD(
                self.classifier.parameters(),
                lr = self.config.model_train_classifier_learning_rate,
                momentum = self.config.model_train_classifier_momentum,
                weight_decay = self.config.model_train_classifier_weight_decay
            )

        self.loss_function = CrossEntropyLoss()
    
    def predict(self, entries : list[str], eval_grad : bool = False):
        if eval_grad:
            embeddings : Tensor = self.embedder(entries) # shape(batch x config.embeddingmodel_dim)
            logits = self.classifier(embeddings)
        else :
            with no_grad():
                embeddings : Tensor = self.embedder(entries) # shape(batch x config.embeddingmodel_dim)
                logits = self.classifier(embeddings)
        return logits

    def train_loop(self, loader : DataLoader, epoch : int) -> None: 
        # UPGRADE the metrics are averages of averages, which is very bad
        metrics : dict[str:float] = {"f1" : 0, "roc_auc" : 0, "accuracy" : 0}
        for batch in tqdm(loader, 
                          desc = "Training loop", leave = False, position = 1):
            self.optimizer_classifier.zero_grad()
            self.optimizer_embedding.zero_grad()

            prediction_logits = self.predict(batch["text"], eval_grad = True)
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"])
            self.history.append_loss_train(epoch, loss.item())
            loss.backward()

            batch_metrics = self.evaluator(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            )
            metrics = {
                key : metrics[key] + batch_metrics[key] for key in metrics
            }
            self.optimizer_classifier.step()
            self.optimizer_embedding.step()
        metrics = {
            key : metrics[key] / len(loader) for key in metrics
        }
        self.history.append_metrics(epoch, "train", metrics)

    def validation_loop(self, loader : DataLoader, epoch : int) -> None:
        # UPGRADE the metrics are averages of averages, which is very bad
        loss_value : float = 0
        metrics : dict[str:float] = {"f1" : 0, "roc_auc" : 0, "accuracy" : 0}
        for batch in tqdm(loader, 
                          desc = "Testing loop", leave = False, position = 1):
            prediction_logits = self.predict(batch["text"], eval_grad = False)
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            ) #FIXME reduction
            loss_value += loss.item()
            batch_metrics = self.evaluator(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            )
            metrics = {
                key : metrics[key] + batch_metrics[key] for key in metrics
            }
        loss_value = loss_value / len(loader.dataset)
        self.history.append_loss_validation(epoch, loss_value = loss_value)
        metrics = {
            key : metrics[key] / len(loader) for key in metrics
        }
        self.history.append_metrics(epoch, "validation", metrics)

    
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