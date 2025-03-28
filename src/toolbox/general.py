
# TODO Réaliser un bandeau
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import DatasetDict, Dataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn import Sigmoid
from transformers import EvalPrediction
# Native
import os

# FUNCTIONS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def storage_options():
    return {
        'client_kwargs': {'endpoint_url': 'https://minio-simple.lab.groupe-genes.fr'},
        'key': os.environ["AWS_ACCESS_KEY_ID"],
        'secret': os.environ["AWS_SECRET_ACCESS_KEY"],
        'token': os.environ["AWS_SESSION_TOKEN"]
    }


def split_test_train_valid(dataset : Dataset, proportion_train : float = 0.7,
    proportion_test : float = None, proportion_valid : float = None,
    shuffle : bool = True, seed : int = 42, print_proportions : bool = False
    ) -> DatasetDict:
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

    if print_proportions : print_datasetdict_proportions(ds)
    return ds

def print_datasetdict_proportions(ds : DatasetDict):
    def proportion(name):
        return int(100 * len(ds[name]) / sum(ds.num_rows.values()))
    print("Répartition des datasets : ")
    print(f'| {"Dataset":^15}|{"Taille":^10}|{"Proportion":<7} (%)|')
    print("-" * 43)
    print(f'| {"Train":<15}|{len(ds["train"]):^10}|{proportion("train"):^14}|')
    print(f'| {"Validation":<15}|{len(ds["validation"]):^10}|{proportion("validation"):^14}|')
    print(f'| {"Test":<15}|{len(ds["test"]):^10}|{proportion("test"):^14}| ')

def get_label_label2id_id2label(ds, print_labels : bool = False):
    # because we are saving data as tensor, the list(set()) returns weird result 
    # so we need to convert the tensor to a list
    LABEL : list[str] = list(set(ds["leaning"].tolist())); 
    n_labels : int = len(LABEL)
    ID2LABEL : dict[int:str] = {i : cat for i,cat in enumerate(LABEL)}
    LABEL2ID : dict[str:int] = {cat:i for i,cat in enumerate(LABEL)}
    if print_labels : print("Categories : " + ", ".join([cat for cat in LABEL]))
    return LABEL, n_labels, ID2LABEL, LABEL2ID

def create_target(batch_leaning : Tensor, n_labels : int,
                  local_device : str = "cpu", dtype = bool) -> Tensor:
    return Tensor(
        [
            [j == logit for j in range(n_labels)]
            for logit in batch_leaning.to(local_device, non_blocking=True)
        ]
    ).to(device = local_device, dtype = dtype, non_blocking=True)   

def multi_label_metrics(results_matrix, labels : Tensor, threshold : float = 0.5
                        ) -> dict:
    '''Taking a results matrix (batch_size x num_labels), the function (with a 
    threshold) associates labels to the results => y_pred
    From this y_pred matrix, evaluate the f1_micro, roc_auc and accuracy metrics
    '''
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = Sigmoid()
    probs = sigmoid(Tensor(results_matrix))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    return {'f1': f1_micro_average,
             'roc_auc': roc_auc,
             'accuracy': accuracy}

def compute_metrics(model_output: EvalPrediction):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions

    metrics = multi_label_metrics(results_matrix=results_matrix, 
        labels=model_output.label_ids)
    return metrics