from time import time
import pandas as pd
from torch import Tensor, sigmoid, no_grad
from torch.cuda import synchronize as torch_synchronize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from .general import create_target
from .metrics import classifier_metrics
class Trainer:
    def __init__(self, model, optimizer, loss_fn, dev_mode : bool = True):
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.dev_mode = dev_mode
        self.history : dict[list[dict]] = {
            "train" : [],
            "validation" : []
        }

    def loop(self, batch_iterable : DataLoader, train_mode : bool) -> None :
        mode : str = "train" if train_mode else "validation"
        iteration_start : float = time()
        sum_loss : float = 0.
        metrics_averaged : dict[str:float] = {
            'f1': 0.,'roc_auc': 0.,'accuracy' : 0.
        }
        # batch_iterable is on CPU
        for batch in tqdm(batch_iterable,desc = mode, position = 1, leave = False):
            # Prepare the loop
            if train_mode: self.__optimizer.zero_grad()

            # Proceed to the classification
            # batch["embedding"].shape = (batch_size, entry_length, model_embeding dimension)
            embeddings = batch["embedding"][:,0,:].\
                            view(-1, self.__model.in_features).\
                            to(
                                device = self.__model.device, 
                                dtype = self.__model.dtype,
                                non_blocking=True
                            )
            logits : Tensor = self.__model(embeddings) # (batch_size, n_labels) ON DEVICE
            probabilities : Tensor = sigmoid(logits) # (batch_size, n_labels) ON DEVICE
            
            # Evaluate the loss ON DEVICE
            loss = self.__loss_fn(
                probabilities,
                create_target(
                    batch["leaning"], 
                    n_labels = self.__model.out_features,
                    local_device = self.__model.device, 
                    dtype = self.__model.dtype
                )
            )
            sum_loss += loss.detach().cpu().item() # ON CPU

            # Evaluate the metrics ON CPU
            metrics : dict[str:float] = classifier_metrics(
                create_target(
                    batch["leaning"],
                    n_labels = self.__model.out_features,
                    local_device = "cpu"
                ),probabilities.detach().cpu()
            )
            # Save the metrics
            for key in metrics : metrics_averaged[key] += metrics[key]

            # Back propagation
            if train_mode: loss.backward()

            # optimizer step
            if train_mode: self.__optimizer.step()
                
        self.history[mode].append({
            "iteration_time" : time() - iteration_start,
            "loss" : sum_loss / len(batch_iterable),
            **{
                key : metrics_averaged[key] / len(batch_iterable)
                for key in metrics_averaged
            }
        })
    
    def train(self,train_iterable : DataLoader, validation_iterable : DataLoader,
              PRS : dict) -> None:
        for epoch in tqdm(range(PRS["n_epoch"]), desc = "Epoch", position = 0):
            self.__model.train() # training mode
            self.loop(train_iterable, train_mode = True) 
            if self.__model.device == "cuda": torch_synchronize()

            self.__model.eval() # evaluation mode
            with no_grad():
                self.loop(validation_iterable, train_mode = False)
                if self.__model.device == "cuda": torch_synchronize()

    def save_history(self, filename : str) -> None:
        new_df = []
        for mode in self.history.keys():
            for i in range(len(self.history[mode])):
                new_df.append({
                    "mode" : mode,
                    "epoch" : i,
                    **self.history[mode][i]
                })
        pd.DataFrame(new_df).to_csv(filename, index = False)