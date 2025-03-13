# TODO Réaliser un bandeau
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import load_from_disk, DatasetDict, Dataset
from pandas import read_csv, DataFrame
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from time import time
from torch import float32, Tensor, sigmoid, empty, no_grad
from torch import where as torch_where
from torch.cuda import is_available as gpu_available
from torch.cuda import synchronize as torch_synchronize
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding
# Native
import json

# Custom
from toolbox import (
    storage_options, split_test_train_valid,
    get_label_label2id_id2label
)
from toolbox.IdeologySentenceClassifier import IdeologySentenceClassifier
from toolbox.Trainer import Trainer
# PARAMETERS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
device = "cuda" if gpu_available() else "cpu"
float_dtype = float32


with open("configs/316_ideology_sentence.json", "r") as file : 
    PRS : dict = json.load(file)

# Load Dataset - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
ds_embed : Dataset = load_from_disk(
    PRS["filename_open_embed"],storage_options = storage_options()
) # keys : ['sentence', 'leaning', 'attention_mask', 'input_ids', 'embedding']
LABEL, n_labels, ID2LABEL, LABEL2ID = get_label_label2id_id2label(ds_embed)

# >>> Split dataframe :
ds : DatasetDict = split_test_train_valid(ds_embed, print_proportions = True) 
print(">>> Load Dataset - Done\n")

# Load custom classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
isc = IdeologySentenceClassifier(
    in_features = PRS["model"]["dim"],out_features = n_labels, 
    hidden_layers = None, hidden_layers_size = None,
    device = device, dtype = float_dtype)
print(isc)
print(">>> Load custom classifier - Done\n")

# Load custom trainer and train - - - - - - - - - - - - - - - - - - - - - - - - 
trainer = Trainer(isc,
        optimizer=Adam(isc.parameters(), **PRS["optimizer"]),
        loss_fn = BCEWithLogitsLoss(),
        dev_mode = True
)
train_iterable : DataLoader = DataLoader(ds["train"],
                            batch_size = PRS["batch_size"], **PRS["DataLoader"])
validation_iterable : DataLoader = DataLoader(ds["train"],
                            batch_size = PRS["batch_size"], **PRS["DataLoader"])

print(">>> Start training\n")
trainer.train(train_iterable,validation_iterable,PRS)
print(">>> Training - Done\n")


