# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import Dataset
from numpy import inf
from torch import Tensor, no_grad, load, save
from torch.cuda import synchronize, ipc_collect, empty_cache
from torch.optim import Adam, SGD, AdamW, Adamax
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
# TODO add a scheduler
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
                **self.config.model_train_embedding_adam_parameters
            )
        elif self.config.model_train_embedding_optimizer == "AdamW":
            self.optimizer_embedding = AdamW(
                self.embedder.model.parameters(),
                **self.config.model_train_embedding_adamw_parameters
            )
        elif self.config.model_train_embedding_optimizer == "Adamax":
            self.optimizer_embedding = Adamax(
                self.embedder.model.parameters(),
                **self.config.model_train_embedding_adamw_parameters
            )

        if self.config.model_train_classifier_optimizer == "SGD":
            self.optimizer_classifier = SGD(
                self.classifier.parameters(),
                **self.config.model_train_classifier_sgd_parameters
            )
        elif self.config.model_train_classifier_optimizer == "Adam":
            self.optimizer_classifier = Adam(
                self.classifier.parameters(),
                **self.config.model_train_classifier_adam_parameters
            )
        elif self.config.model_train_classifier_optimizer == "AdamW":
            self.optimizer_classifier = AdamW(
                self.classifier.parameters(),
                **self.config.model_train_classifier_adamw_parameters
            )
        elif self.config.model_train_classifier_optimizer == "Adamax":
            self.optimizer_classifier = Adamax(
                self.classifier.parameters(),
                **self.config.model_train_classifier_adamax_parameters
            )

        self.loss_function = CrossEntropyLoss(**self.config.model_loss_parameters)
        self.best_embedder = None
        self.best_classifier = None
    
    def load_best(self) : 
        self.best_embedder = CustomEmbedder(self.config)
        self.best_embedder.load_from_disk(self.config.embeddingmodel_save_filename)
        self.best_classifier = CustomClassifier(self.config).to(device = self.config.device)
        self.best_classifier.load_state_dict(
            load(self.config.classifier_save_filename, weights_only=True)
        )

    def predict(self, entries : list[str], eval_grad : bool = False,
                use_best : bool = False) -> Tensor:
        can_use_best_embedder = not(self.best_embedder is None)
        can_use_best_classifier = not(self.best_embedder is None)

        embedder = self.best_embedder if use_best&can_use_best_embedder else self.embedder
        classifier = self.best_classifier if use_best&can_use_best_classifier else self.classifier
        
        if eval_grad:
            embeddings : Tensor = embedder(entries) # shape(batch x config.embeddingmodel_dim)
            logits = classifier(embeddings)
        else :
            with no_grad():
                embeddings : Tensor = embedder(entries) # shape(batch x config.embeddingmodel_dim)
                logits = classifier(embeddings)
        return logits

    def train_loop(self, loader : DataLoader, epoch : int) -> None: 
        log_probs : list[list[float]] = []
        labels : list[int] = []
        loss_value = 0
        for batch in tqdm(loader, 
                          desc = "Training loop", leave = False, position = 2):
            self.optimizer_classifier.zero_grad()
            self.optimizer_embedding.zero_grad()

            prediction_logits = self.predict(batch["text"], eval_grad = True)
            
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"])
            loss.backward()

            loss_value += loss.item()
            log_probs.extend(prediction_logits.to(device = "cpu").tolist())
            labels.extend(batch["label"])

            self.optimizer_classifier.step()
            self.optimizer_embedding.step()
        
        metrics = self.evaluator(log_probs,labels)
        confusion_matrix = self.evaluator.confusion_matrix(log_probs,labels)
        for idlabel in confusion_matrix.keys() : 
            metrics[f"f1_{self.config.dataset_id2label[idlabel]}"] = self.evaluator.f1(idlabel)

        self.history.append_loss_train(epoch, loss_value / len(loader.dataset))
        self.history.append_metrics(epoch, "train", metrics)
        self.history.append_confusion_matrix(epoch, confusion_matrix,
            tag = "train", id2label = self.config.dataset_id2label)

    def validation_loop(self, loader : DataLoader, epoch : int) -> None:
        loss_value : float = 0
        log_probs : list[list[float]] = []
        labels : list[int] = []
        for batch in tqdm(loader, 
                          desc = "Validation loop", leave = False, position = 2):
            prediction_logits = self.predict(batch["text"], eval_grad = False)
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            )

            loss_value += loss.item()
            log_probs.extend(prediction_logits.to(device = "cpu").tolist())
            labels.extend(batch["label"])

        metrics = self.evaluator(log_probs,labels)
        confusion_matrix = self.evaluator.confusion_matrix(log_probs,labels)
        for idlabel in confusion_matrix.keys() : 
            metrics[f"f1_{self.config.dataset_id2label[idlabel]}"] = self.evaluator.f1(idlabel)

        self.history.append_loss_validation(epoch,loss_value / len(loader.dataset))
        self.history.append_metrics(epoch, "validation", metrics)
        self.history.append_confusion_matrix(epoch, confusion_matrix,
                tag = "validation", id2label = self.config.dataset_id2label)
        
    def test_loop(self, loader : DataLoader) -> None:
        loss_value : float = 0
        log_probs : list[list[float]] = []
        labels : list[int] = []
        for batch in tqdm(loader, 
                          desc = "Test loop", leave = False, position = 2):
            prediction_logits = self.predict(batch["text"], eval_grad = False)
            loss = self.loss_function(
                prediction_logits.to(device = "cpu"), 
                batch["label"]
            )

            loss_value += loss.item()
            log_probs.extend(prediction_logits.to(device = "cpu").tolist())
            labels.extend(batch["label"])

        metrics = self.evaluator(log_probs,labels)
        confusion_matrix = self.evaluator.confusion_matrix(log_probs,labels)
        for idlabel in confusion_matrix.keys() : 
            metrics[f"f1_{self.config.dataset_id2label[idlabel]}"] = self.evaluator.f1(idlabel)
            
        self.history.append_metrics(-1, "test", metrics)
        self.history.append_confusion_matrix(-1, confusion_matrix,
                tag = "test", id2label = self.config.dataset_id2label)
    
    def train(
            self, train_dataset : Dataset, validation_dataset : Dataset, 
            callback_function = lambda epoch : None, callback_parameters : dict = {}
            ) -> None:
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

            if (self.history.validation_loss[-1]["loss_value"] <= lowest_validation_loss)&\
                self.config.model_save_best_model: 
                lowest_validation_loss = self.history.validation_loss[-1]["loss_value"]
                self.embedder.save_to_disk(self.config.embeddingmodel_save_filename)
                save(self.classifier.state_dict(), self.config.classifier_save_filename)
            
            callback_function(epoch = epoch, **callback_parameters)

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