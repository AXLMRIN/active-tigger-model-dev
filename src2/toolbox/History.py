# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
import pandas as pd
import plotly.graph_objects as go
# Native
import os
import shutil
# Custom

# CLASS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class History:
    def __init__(self):
        self.train_loss_per_epoch : dict[int:list[float]] = {}
        self.train_loss_global : list[float] = []
        self.validation_loss : dict[int:float] = {}
        self.metrics_save : dict = {
            "epoch" : [],
            "loop" : [],
            "f1" : [],
            "roc_auc" : [],
            "accuracy" : []
        }

    def append_loss_train(self, epoch : int, loss_value : float) -> None:
        self.train_loss_global.append(loss_value)
        
        if epoch not in self.train_loss_per_epoch:
            self.train_loss_per_epoch[epoch] = []
        self.train_loss_per_epoch[epoch].append(loss_value)
    
    def append_loss_validation(self, epoch : int, loss_value : float) -> None :
        self.validation_loss[epoch] = loss_value

    def append_metrics(self,epoch : int, loop : str, 
                       metrics : dict[str:float]) -> None:
        self.metrics_save["epoch"].append(epoch)
        self.metrics_save["loop"].append(loop)
        self.metrics_save["f1"].append(metrics["f1"])
        self.metrics_save["roc_auc"].append(metrics["roc_auc"])
        self.metrics_save["accuracy"].append(metrics["accuracy"])

    def plot_loss_train(self):
        # UPGRADE MAKE IT BETTER
        go.Figure(
            data = [
                go.Scatter(
                    x = [i for i in range(len(self.train_loss_global))],
                    y = self.train_loss_global,
                    name = "Train Loss"
                )
            ],
        ).write_image("train_loss.png")

    def plot_all(self):
        self.plot_loss_train()
    def save_all(self, foldername):
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
        os.makedirs(foldername)

        pd.DataFrame(self.metrics_save).to_csv(foldername + "/metrics_save.csv")
        pd.DataFrame(self.train_loss_global).to_csv(foldername + "/loss_train.csv")
        pd.DataFrame(self.validation_loss).to_csv(foldername + "/validation_loss.csv")

    def __str__(self) -> str:
        return (
            "History object"
        )