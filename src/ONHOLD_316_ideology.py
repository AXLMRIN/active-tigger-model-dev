# TODO Réaliser un bandeau

# DEBUG FIXME
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import load_dataset, DatasetDict
from pandas import read_csv, DataFrame
from time import time
from torch import float32, Tensor, sigmoid, empty, no_grad
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

# Custom
from src.toolbox.ABORTED_IdeologySentenceClassifier import IdeologySentenceClassifier
from toolbox.metrics import classifier_metrics

# PARAMETERS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
att_implementation : str = "sdpa"
# TODO Demander pour flash_attention_2
device = "cuda" if gpu_available() else "cpu"
float_dtype = float32

# Load parameters saved as json #TODO is it really necessary ??
import json
with open("configs/316_ideology_sentence.json", "r") as file : 
    PRS : dict = json.load(file)


# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# Load Dataset - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#TODO Clean this is litteraly USELESS
df = read_csv(PRS["filename_open"]).\
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

ds_original = load_dataset("csv", data_files = {"whole" :PRS["filename_open"]})["whole"]
# split the dateaset into (test) and (train, validation)
ds_temp = ds_original.train_test_split(test_size = 0.15,shuffle = True, seed = PRS["seed"])
# split the dataset into (train) and (validation)
ds_temp2 = ds_temp["train"].train_test_split(train_size = 0.82,shuffle = True, seed = PRS["seed"])

ds = DatasetDict({
    "train" : ds_temp2["train"],
    "validation" : ds_temp2["test"],
    "test" : ds_temp["test"],

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
base_model = ModernBertModel.from_pretrained(PRS["model"]["name"],
                attn_implementation = att_implementation,
                num_labels = n_labels,
                id2label = ID2LABEL,
                label2id = LABEL2ID).\
                to(device)
tokenizer = AutoTokenizer.from_pretrained(PRS["model"]["name"])
# Load custom classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
isc = IdeologySentenceClassifier(
    in_features = PRS["model"]["dim"],out_features = n_labels, 
    hidden_layers = None, hidden_layers_size = None,
    device = device, dtype = float_dtype)
print(isc)
print(">>> Load custom classifier - Done")

# Create the train_loop and eval_loop functions - - - - - - - - - - - - - - - - 
# Create the instances of the loss function and the optimizer
loss_fn = BCEWithLogitsLoss()
optimizer = Adam(isc.parameters(), lr = 1e-2)

# Function creation
def create_target(batch_leaning : Tensor, local_device : str = device,
                  dtype = bool) -> Tensor:
    return Tensor(
            [
                [j == logit for j in range(n_labels)]
                for logit in batch_leaning.to(local_device, non_blocking=True)
            ]
        ).to(device = local_device, dtype = dtype, non_blocking=True)   

def train_loop(batch_iterable : DataLoader) -> list[dict] :
    iteration_start : float = time()
    sum_loss : float = 0.
    metrics_averaged : dict[str:float] = {
        'f1': 0.,'roc_auc': 0.,'accuracy' : 0.
    }
    # batch_iterable is on CPU
    for batch in tqdm(batch_iterable):
        # Prepare the loop
        optimizer.zero_grad()

        # Encode the input ON DEVICE
        encoded : BatchEncoding = tokenizer(batch["sentence"], **PRS["tokenizing"])
        # Move to DEVICE to use in base_model : 
        encoded.to(device = device, non_blocking = True)
        #The BaseModelOutput.last_hidden_state is a torch.Tensor of dimension
        # (batch_size, seq_lengthm, embedding_dim)
        # because it holds the embedding of the batch_size sentences, each of lenght 
        # seq_lengthm and the dimension of each embedding is embedding_dim (=768)
        # Although, we are classifying on the [CLS] token, so we only keep the first item
        # (Hence the [:,0,:]) and reshaping it to (batch_size, embedding_dim) for the
        # ics to accept it
        # Embed on DEVICE
        embeddings : BaseModelOutput = base_model(**encoded).\
                last_hidden_state[:,0,:].\
                view(-1,PRS["model"]["dim"] )
        # Proceed to the classification
        logits : Tensor = isc(embeddings) # (batch_size, n_labels) ON DEVICE
        probabilities : Tensor = sigmoid(logits) # (batch_size, n_labels) ON DEVICE
        
        # Evaluate the loss ON DEVICE
        loss = loss_fn(
            probabilities,
            create_target(batch["leaning"], local_device = device, dtype = float_dtype)
        )
        sum_loss += loss.detach().cpu().item() # ON CPU

        # Evaluate the metrics ON CPU
        metrics : dict[str:float] = classifier_metrics(
            create_target(batch["leaning"], local_device = "cpu"),
            probabilities.detach().cpu())
        # Save the metrics
        for key in metrics : metrics_averaged[key] += metrics[key]

        # Back propagation
        loss.backward()

        # optimizer step
        optimizer.step()
        break

    return {
        "iteration_time" : time() - iteration_start,
        "loss" : sum_loss / len(batch_iterable),
        **{
            key : metrics_averaged[key] / len(batch_iterable)
            for key in metrics_averaged
        }
    }
        
def eval_loop(batch_iterable : DataLoader):
    iteration_start : float = time()
    sum_loss : float = 0.
    metrics_averaged : dict[str:float] = {
        'f1': 0.,'roc_auc': 0.,'accuracy' : 0.
    }
    # batch_iterable is on CPU
    with no_grad():
        for batch in tqdm(batch_iterable):
            # Encode the input ON DEVICE
            encoded : BatchEncoding = tokenizer(batch["sentence"], **PRS["tokenizing"])
            # Move to DEVICE to use in base_model : 
            encoded.to(device = device, non_blocking = True)
            # Embed on DEVICE
            embeddings : BaseModelOutput = base_model(**encoded).\
                    last_hidden_state[:,0,:].\
                    view(-1,PRS["model"]["dim"] )
            # Proceed to the classification
            logits : Tensor = isc(embeddings) # (batch_size, n_labels) ON DEVICE
            probabilities : Tensor = sigmoid(logits) # (batch_size, n_labels) ON DEVICE
            
            # Evaluate the loss ON DEVICE
            loss = loss_fn(
                probabilities,
                create_target(batch["leaning"], local_device = device, dtype = float_dtype)
            )
            sum_loss += loss.detach().cpu().item() # ON CPU

            # Evaluate the metrics ON CPU
            metrics : dict[str:float] = classifier_metrics(
                create_target(batch["leaning"], local_device = "cpu"),
                probabilities.detach().cpu())
            # Save the metrics
            for key in metrics : metrics_averaged[key] += metrics[key]

            break
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
train_batch_iterable : DataLoader = DataLoader(ds["train"],
                        batch_size = PRS["batch_size"], **PRS["DataLoader"])
validation_batch_iterable : DataLoader = DataLoader(ds["validation"],
                        batch_size = PRS["batch_size"], **PRS["DataLoader"])

# Record of all the metrics (time, iteration, train_loss, f1, accuracy and auc_roc)
# for all epochs
train_book : dict[int:list[dict]] = {}
validation_book : dict[int:list[dict]] = {}

# Training loop :
for epoch in range(PRS["n_epoch"]):
    print(f"Epoch : {epoch} / {PRS["n_epoch"]}")
    isc.train()
    record_train = train_loop(train_batch_iterable) 
    train_book[epoch] = record_train
    torch_synchronize()

    isc.eval()
    record_eval = eval_loop(validation_batch_iterable)
    validation_book[epoch] = record_eval
    torch_synchronize()


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
DataFrame(df).to_csv(PRS["csv_train_save"], index= False)
print(">>> Train and saving - Done")