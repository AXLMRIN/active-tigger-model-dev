# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import Dataset
from numpy import inf
from torch import Tensor, no_grad, load, save
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

        self.loss_function_train = CrossEntropyLoss(reduction = "mean")
        self.loss_function_validation = CrossEntropyLoss(reduction = "sum")
        self.best_embedder = None
        self.best_classifier = None
    
    def load_best(self) : 
        self.best_embedder.load_from_disk(self.config.embeddingmodel_save_filename)
        self.best_classifier = CustomClassifier(self.config)
        self.best_classifier.load_state_dict(
            load(self.config.classifier_save_filename, weights_only=True)
        )

    def predict(self, entries : list[str], eval_grad : bool = False,
                use_best : bool = False):
        can_use_best_embedder = not(self.best_embedder is None)
        can_use_best_classifier = not(self.best_embedder is None)

        embedder = self.best_embedder if use_best&can_use_best_embedder else self.embedder
        classifier = self.best_classifier if use_best&can_use_best_embedder else self.classifier
        
        if eval_grad:
            embeddings : Tensor = embedder(entries) # shape(batch x config.embeddingmodel_dim)
            logits = classifier(embeddings)
        else :
            with no_grad():
                embeddings : Tensor = embedder(entries) # shape(batch x config.embeddingmodel_dim)
                logits = classifier(embeddings)
        return logits

    def train_loop(self, loader : DataLoader, epoch : int) -> None: 
        # UPGRADE the metrics are averages of averages, which is very bad
        metrics : dict[str:float] = {"f1" : 0, "roc_auc" : 0, "accuracy" : 0}
        for batch in tqdm(loader, 
                          desc = "Training loop", leave = False, position = 2):
            self.optimizer_classifier.zero_grad()
            self.optimizer_embedding.zero_grad()

            prediction_logits = self.predict(batch["text"], eval_grad = True)
            loss = self.loss_function_train(
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
                          desc = "Validation loop", leave = False, position = 2):
            prediction_logits = self.predict(batch["text"], eval_grad = False)
            loss = self.loss_function_validation(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            )
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
        lowest_validation_loss = inf
        for epoch in tqdm(range(self.config.model_train_n_epoch),
                          desc = "Train Epoch", leave = False, position = 1):
            self.embedder.train()
            self.classifier.train()
            self.train_loop(train_loader, epoch)
            self.embedder.eval()
            self.classifier.eval()
            self.validation_loop(validation_loader, epoch)
            
            if self.history.validation_loss[-1] <= lowest_validation_loss : 
                lowest_validation_loss = self.history.validation_loss[-1]
                self.embedder.save_to_disk(self.config.embeddingmodel_save_filename)
                save(self.classifier.state_dict(), self.config.classifier_save_filename)

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