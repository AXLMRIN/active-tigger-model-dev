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
        self.threshold = threshold
        self.n_label = n_label
        self.device = "cpu"

    def create_target(self,labels : list[int]) -> Tensor:
        return Tensor(
            [
                [col == logit for col in range(self.n_label)]
                for logit in labels
            ]
        ).to(device = self.device, dtype = bool, non_blocking=True)   
    
    def __call__(self, result_logits : Tensor, labels : list[int]) -> dict:
        """EVERYTHING HAPPENS ON CPU
        labels are only the id of the labels"""
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = Sigmoid()
        probs = sigmoid(result_logits)
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(log_probs.shape)
        y_pred[np.where(log_probs >= self.log_threshold)] = 1
        # finally, compute metrics
        y_true = self.create_target(labels)
        f1_micro_average = f1_score(
            y_true=y_true, 
            y_pred=y_pred, 
            average='micro'
        )
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro').item()
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        return {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}