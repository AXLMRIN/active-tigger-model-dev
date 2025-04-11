# IMPORTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Third parties
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn import Sigmoid
# Native

# Custom
from .Config import Config

# TYPES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def split_test_train_valid(dataset : Dataset, proportion_train : float = 0.7,
    proportion_test : float = None, proportion_valid : float = None,
    shuffle : bool = True, seed : int = 42) -> DatasetDict:
    if (proportion_test is None) & (proportion_valid is None):
        proportion_test = (1 - proportion_train) / 2
        proportion_valid = proportion_test
    elif (proportion_test is None) : 
        proportion_test = 1 - proportion_train - proportion_valid
    elif (proportion_valid is None) :
        proportion_valid = 1 - proportion_train - proportion_test
    try:
        assert(proportion_train + proportion_test + proportion_valid == 1)
    except:
        print((
            "WARNING Wrong entry. Your entries :\n"
            "\t - proportion_train : {proportion_train}\n"
            "\t - proportion_test : {proportion_test}\n"
            "\t - proportion_valid : {proportion_valid}\n"
            "\n"
            "By default we use the tuple (0.7,0.15,0.15)"
        ))
        proportion_train, proportion_test, proportion_valid = 0.7,0.15,0.15

    ds_temp = dataset.train_test_split(test_size = proportion_test, 
                shuffle=shuffle, seed = seed)
    ds_temp2 = ds_temp["train"].train_test_split(
                        train_size = proportion_train / (1- proportion_test),
                        shuffle = shuffle, seed = seed
    )
    ds = DatasetDict({
        "train" : ds_temp2["train"],
        "validation" : ds_temp2["test"],
        "test" : ds_temp["test"],
    })

    return ds

class Evaluator:
    def __init__(self, n_label : int, threshold : float = 0.5):
        self.log_threshold = np.log(threshold) # UNUSED FIXME
        self.n_label = n_label
        self.device = "cpu"
        self.CM = None

    def create_target(self,labels : list[int]) -> Tensor:
        return Tensor(
            [
                [col == logit for col in range(self.n_label)]
                for logit in labels
            ]
        ).to(device = self.device, dtype = bool, non_blocking=True)   
    
    def __call__(self, log_probs : Tensor, labels : list[int]) -> dict:
        """EVERYTHING HAPPENS ON CPU
        labels are only the id of the labels
        The results are coming out of the log_softmax activation function
        """
        # next, use the max logit to make the prediction
        y_pred = np.array([
            [element == max(row) for element in row]
            for row in log_probs
        ])
        # y_pred = np.zeros(log_probs.shape)
        # y_pred[np.where(log_probs >= self.log_threshold)] = 1
        # finally, compute metrics
        y_true = self.create_target(labels)
        f1_micro_average = f1_score(
            y_true=y_true, 
            y_pred=y_pred, 
            average='micro'
        )
        f1_macro_average = f1_score(
            y_true=y_true, 
            y_pred=y_pred, 
            average='macro'
        )
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro').item()
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        return {'f1_micro': f1_micro_average,
                "f1_macro": f1_macro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
    
    def confusion_matrix(self, log_probs : Tensor, labels_true : list[int]) -> dict:
        """EVERYTHING HAPPENS ON CPU
        labels are only the id of the labels
        The results are coming out of the log_softmax activation function

            PREDICTED
        T   x | x | x | x
        R   x | x | x | x
        U   x | x | x | x
        E   x | x | x | x
        """

        labels_pred = np.array([
            np.argmax(row) for row in log_probs
        ])
        labels_true = [int(label) for label in labels_true]
        confusion_matrix = {
            i : {j : 0 for j in range(self.n_label)} 
            for i in range(self.n_label)
        }
        for pred, true in zip(labels_pred, labels_true) :
            confusion_matrix[true][pred] += 1
        
        self.CM : np.ndarray= np.array([
            list(confusion_matrix[true].values())
            for true in confusion_matrix
        ])

        return confusion_matrix
    
    def f1(self, idlabel) -> float: 
        if self.CM is None :return -1
        elif self.CM[idlabel,:].sum().item() == 0 : return -2
        elif self.CM[:,idlabel].sum().item() == 0 : return -3
        elif self.CM[idlabel,idlabel].item() == 0 : return -4

        acc = self.CM[idlabel,idlabel].item() /\
              self.CM[idlabel,:].sum().item()
        prec = self.CM[idlabel,idlabel].item() /\
               self.CM[:,idlabel].sum().item()
        return 2 * acc * prec / (prec + acc)
        