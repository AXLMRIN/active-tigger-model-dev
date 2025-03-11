# TODO Réaliser un bandeau

# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import load_dataset, DatasetDict
from pandas import read_csv, DataFrame
from time import time
from torch import float32, Tensor, sigmoid, empty, no_grad
from torch.cuda import is_available as gpu_available
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding
# Native

# Custom
from toolbox.IdeologySentenceClassifier import IdeologySentenceClassifier

# PARAMETERS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
FILENAME : str = "data/316_ideological_book_corpus/ibc.csv"
train_record_save_filename : str = "316_ideological_book_corpus-IdeologySentenceClassifier-train.csv"
seed : int = 42

model_name : str = "answerdotai/ModernBERT-base"; embedding_dim : int = 768
att_implementation : str = "sdpa"
# TODO Demander pour flash_attention_2
device = "cuda" if gpu_available() else "cpu"
float_dtype = float32

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
def classifier_metrics(y_true : Tensor, y_pred : Tensor) -> dict[str:float]:
    y_true = y_true.to(bool, copy = True)
    y_pred = y_pred.to(bool, copy = True)
    
    return {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'roc_auc': roc_auc_score(y_true, y_pred, average = 'micro').item(),
        'accuracy': accuracy_score(y_true, y_pred)
    }

# from torch.nn import Threshold; threshold : Threshold = Threshold(0.4,0)
from torch import where as torch_where
def threshold(probabilities : Tensor, thresh_value : float = 0.4) -> Tensor :
    return torch_where(probabilities > 0.4,Tensor([1.0]).to(device),
                       Tensor([0.0]).to(device)).to(float_dtype)

n_epoch : int = 2

# Load parameters saved as json #TODO is it really necessary ??
import json
with open("configs/316_ideology_sentence.json", "r") as file : 
    parameters = json.load(file)

# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# Load Dataset - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#TODO Clean this is litteraly USELESS
df = read_csv(FILENAME).\
    astype({
        "sentence" : "str",
        "leaning" : "str"
    }).\
    rename({
        "sentence" : "in",
        "leaning" : "out"
    }, axis = 1)
LABEL : list[str] = list(set(df["out"])); n_labels : int = len(LABEL)
ID2LABEL : dict[int:str] = {i : cat for i,cat in enumerate(LABEL)}
LABEL2ID : dict[str:int] = {cat:i for i,cat in enumerate(LABEL)}
print("Categories : " + ", ".join([cat for cat in LABEL]),"\n")
class_size = df.groupby("out").size()
def dot2f(x):
    return int(x * 100) / 100
def div(col1, col2) : 
    return dot2f(class_size[col1] / class_size[col2])

print(f'| {"Label":<15} || {"Amount":<10} || {"Conservative":<15} | {"Liberal":<15} | {"Neutral" : <15} |')
print("-" * 88)
print(f'| {"Conservative":<15} || {class_size["Conservative"]:<10} || {div("Conservative","Conservative"):<15} | {div("Liberal","Conservative"):<15} | {div("Neutral","Conservative"): <15} |')
print(f'| {"Liberal":<15} || {class_size["Liberal"]:<10} || {div("Conservative","Liberal"):<15} | {div("Liberal","Liberal"):<15} | {div("Neutral","Liberal"): <15} |')
print(f'| {"Neutral":<15} || {class_size["Neutral"]:<10} || {div("Conservative","Neutral"):<15} | {div("Liberal","Neutral"):<15} | {div("Neutral","Neutral"): <15} |')
print("-" * 88)

del df, class_size, dot2f, div

ds_original = load_dataset("csv", data_files = {"whole" :FILENAME})["whole"]
# split the dateaset into (test) and (train, validation)
ds_temp = ds_original.train_test_split(test_size = 0.15,shuffle = True, seed = seed)
# split the dataset into (train) and (validation)
ds_temp2 = ds_temp["train"].train_test_split(train_size = 0.82,shuffle = True, seed = seed)

ds = DatasetDict({
    "train" : ds_temp2["train"],
    "validation" : ds_temp2["test"],
    "test" : ds_temp["test"]
})

def proportion(name):
    return int(
        100 * len(ds[name]) / len(ds_original)
    )
print("Répartition des datasets : ")
print(f'| {"Dataset":^15}|{"Taille":^10}|{"Proportion":<7} (%)|')
print("-" * 43)
print(f'| {"Train":<15}|{len(ds["train"]):^10}|{proportion("train"):^14}|')
print(f'| {"Validation":<15}|{len(ds["validation"]):^10}|{proportion("validation"):^14}|')
print(f'| {"Test":<15}|{len(ds["test"]):^10}|{proportion("test"):^14}| ')

print(">>> Load Dataset - Done")
# Preprocess - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def preprocess(row):
    """For now we only uncapitalised the sentences"""
    row["sentence"] = row["sentence"].lower()
    row["leaning"] = LABEL2ID[row["leaning"]]
    return row

ds = ds.map(preprocess)
print(">>> Preprocess - Done")
# Loading the model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
base_model = ModernBertModel.from_pretrained(model_name,
                attn_implementation = att_implementation,
                num_labels = n_labels,
                id2label = ID2LABEL,
                label2id = LABEL2ID)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load custom classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
isc = IdeologySentenceClassifier(
    in_features = embedding_dim,out_features = n_labels, 
    hidden_layers = None, hidden_layers_size = None,
    device = device, dtype = float_dtype)
print(isc)
print(">>> Load custom classifier - Done")

# Create the train_loop and eval_loop functions - - - - - - - - - - - - - - - - 
# Create the instances of the loss function and the optimizer
loss_fn = BCEWithLogitsLoss()
optimizer = Adam(isc.parameters(), lr = 1e-2)

def train_loop(batch_iterable : DataLoader) -> list[dict] :
    iteration_start : float = time()
    sum_loss : float = 0.
    metrics_averaged : dict[str:float] = {
        'f1': 0.,'roc_auc': 0.,'accuracy' : 0.
    }
    for batch in batch_iterable:
        print("Batch device : ",batch.device)#TODELETE
        # Prepare the loop
        optimizer.zero_grad()
        
        # Embedd the input
        encoded : BatchEncoding = tokenizer(batch["sentence"], **parameters["tokenizing"])
        #The BaseModelOutput.last_hidden_state is a torch.Tensor of dimension
        # (batch_size, seq_lengthm, embedding_dim)
        # because it holds the embedding of the batch_size sentences, each of lenght 
        # seq_lengthm and the dimension of each embedding is embedding_dim (=768)
        # Although, we are classifying on the [CLS] token, so we only keep the first item
        # (Hence the [:,0,:]) and reshaping it to (batch_size, embedding_dim) for the
        # ics to accept it
        embeddings : BaseModelOutput = base_model(**encoded).\
                        last_hidden_state[:,0,:].\
                        view(parameters["DataLoader"]["batch_size"],embedding_dim ).\
                        to(float_dtype).to(device)
        # Proceed to the classification
        logits : Tensor = isc(embeddings) # (batch_size, n_labels)
        probabilities : Tensor = sigmoid(logits) # (batch_size, n_labels) NOTE à check parce que c'est pas impossible que le problème vienne d'ici
        prediction : Tensor = threshold(probabilities)
        # Evaluate the loss and metrics
        target : Tensor = Tensor(
            [
                [j == logit for j in range(n_labels)]
                for logit in batch["leaning"]
            ], device=device
        )   
        loss = loss_fn(probabilities, target)
        sum_loss += loss.item()
        metrics : dict[str:float] = classifier_metrics(target, prediction)
        for key in metrics : metrics_averaged[key] += metrics[key]

        # Back propagation
        loss.backward()

        # optimizer step
        optimizer.step()

    return {
        "iteration_time" : time() - iteration_start,
        "loss" : sum_loss / len(batch_iterable),
        **{
            key : metrics_averaged[key] / len(batch_iterable)
            for key in metrics_averaged
        }
    }
        
def eval_loop(batch_iterable):
    # Prepare the loop
    iteration_start : float = time()
    sum_loss : float = 0.
    metrics_averaged : dict[str:float] = {
        'f1': 0.,'roc_auc': 0.,'accuracy' : 0.
    }

    with no_grad():
        for batch in batch_iterable:
            # Embedd the input
            encoded : BatchEncoding = tokenizer(batch["sentence"], 
                                                **parameters["tokenizing"])
            embeddings : BaseModelOutput = base_model(**encoded).\
                            last_hidden_state[:,0,:].\
                            view(parameters["DataLoader"]["batch_size"],embedding_dim ).\
                            to(float_dtype).to(device)

            # Proceed to the classification
            probabilities : Tensor = sigmoid(isc(embeddings)) # (batch_size, n_labels)
            prediction : Tensor = threshold(probabilities)

            # Evaluate the loss and metrics
            target : Tensor = Tensor(
                [
                    [j == logit for j in range(n_labels)]
                    for logit in batch["leaning"]
                ], device=device
            )   
            loss = loss_fn(probabilities, target)
            sum_loss += loss.item()
            metrics : dict[str:float] = classifier_metrics(target, prediction)
            for key in metrics : metrics_averaged[key] += metrics[key]

    return {
        "iteration_time" : time() - iteration_start,
        "loss" : sum_loss / len(batch_iterable),
        **{
            key : metrics_averaged[key] / len(batch_iterable)
            for key in metrics_averaged
        }
    }
print(">>> Loop functions creation - Done")
# Train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
train_batch_iterable : DataLoader = DataLoader(ds["train"], **parameters["DataLoader"])
validation_batch_iterable : DataLoader = DataLoader(ds["validation"], **parameters["DataLoader"])

# Record of all the metrics (time, iteration, train_loss, f1, accuracy and auc_roc)
# for all epochs
train_book : dict[int:list[dict]] = {}
validation_book : dict[int:list[dict]] = {}

# Training loop :
for epoch in tqdm(range(n_epoch)):
    isc.train()
    record_train = train_loop(train_batch_iterable) 
    train_book[epoch] = record_train

    isc.eval()
    record_eval = eval_loop(validation_batch_iterable)
    validation_book[epoch] = record_eval


# >>> Save the results
df : list[dict] = []
for epoch in train_book : 
    df.append({
        "mode" : "train",
        "epoch" : epoch,
        **train_book[epoch]
    })
for epoch in validation_book : 
    df.append({
        "mode" : "validation",
        "epoch" : epoch,
        **validation_book[epoch]
    })
DataFrame(df).to_csv(train_record_save_filename, index= False)
print(">>> Train and saving - Done")