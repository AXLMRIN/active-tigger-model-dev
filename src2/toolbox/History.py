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
        self.train_loss : list[dict[str : int|float]] = []
        self.validation_loss : list[dict[str : int|float]] = []
        self.metrics_save : dict = {
            "epoch" : [],
            "loop" : [],
            "f1" : [],
            "roc_auc" : [],
            "accuracy" : []
        }
        self.confusion_matrix : list[dict[str:int]] = []

    def append_loss_train(self,epoch : int, loss_value : float) -> None:
        self.train_loss.append({
            "epoch" : epoch,
            "loss_value" : loss_value
        })
    
    def append_loss_validation(self, epoch : int, loss_value : float) -> None :
        self.validation_loss.append({
            "epoch" : epoch,
            "loss_value" : loss_value
        })
    
    def append_confusion_matrix(self, epoch : int, confusion_matrix : dict, tag : str,
                         id2label : dict[int:str]) -> None:
        """
              PREDICTED
        T   x | x | x | x
        R   x | x | x | x
        U   x | x | x | x
        E   x | x | x | x
        """
        for idtrue in confusion_matrix :
            self.confusion_matrix.append({
                "epoch" : epoch,
                "tag" : tag,
                "true" : id2label[idtrue],
                **{
                    f"pred_{id2label[idpred]}" : confusion_matrix[idtrue][idpred]
                    for idpred in confusion_matrix[idtrue]
                }
            }) 

    def append_metrics(self,epoch : int, loop : str, 
                       metrics : dict[str:float]) -> None:
        self.metrics_save["epoch"].append(epoch)
        self.metrics_save["loop"].append(loop)
        self.metrics_save["f1"].append(metrics["f1"])
        self.metrics_save["roc_auc"].append(metrics["roc_auc"])
        self.metrics_save["accuracy"].append(metrics["accuracy"])

    def OUTDATED_plot_loss_train(self):
        print("OUTDATED NOTHING RUNS")

    def plot_all(self):
        self.plot_loss_train()
    def save_all(self, foldername):
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
        os.makedirs(foldername)

        pd.DataFrame(self.train_loss).to_csv(foldername + "/train_loss.csv", index = False)
        pd.DataFrame(self.metrics_save).to_csv(foldername + "/metrics_save.csv", index = False)
        pd.DataFrame(self.validation_loss).to_csv(foldername + "/validation_loss.csv", index = False)
        pd.DataFrame(self.confusion_matrix).to_csv(foldername + "/confusion_matrix.csv", index = False)

    def __str__(self) -> str:
        return (
            "History object"
        )