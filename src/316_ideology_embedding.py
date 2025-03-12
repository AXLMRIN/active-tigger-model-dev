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

# Custom
from toolbox import storage_options
from toolbox.IdeologySentenceClassifier import IdeologySentenceClassifier

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
ds : Dataset = Dataset.from_pandas(read_csv(
    PRS["filename_open"], storage_options = storage_options()
))

LABEL : list[str] = list(set(ds["leaning"])); n_labels : int = len(LABEL)
ID2LABEL : dict[int:str] = {i : cat for i,cat in enumerate(LABEL)}
LABEL2ID : dict[str:int] = {cat:i for i,cat in enumerate(LABEL)}
print("Categories : " + ", ".join([cat for cat in LABEL]),"\n")

# TODELETE 
print(("WARNING : you are only selecting a fraction of the real dataset for dev"
       "purposes."))
ds = ds.select(range(0,100))
# --------
print(">>> Load Dataset - Done")
# Preprocess - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def preprocess(batch_of_rows : dict):
    """For now we only uncapitalised the sentences"""
    batch_of_rows["sentence"] = [sentence.lower() 
                                 for sentence in batch_of_rows["sentence"]]
    batch_of_rows["leaning"] = [LABEL2ID[leaning] 
                                for leaning in batch_of_rows["leaning"]]
    return batch_of_rows

ds = ds.map(preprocess, batched = True, batch_size = PRS["batch_size"])
print(">>> Preprocess - Done")
# Loading the model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
base_model = ModernBertModel.from_pretrained(PRS["model"]["name"],
                attn_implementation = att_implementation,
                num_labels = n_labels,
                id2label = ID2LABEL,
                label2id = LABEL2ID).\
                to(device)
tokenizer = AutoTokenizer.from_pretrained(PRS["model"]["name"])

# Tokenize the sentences
def tokenizing_sentences(batch_of_rows : dict):
    tokenized = tokenizer(batch_of_rows["sentence"], **PRS["tokenizing"])
    batch_of_rows["attention_mask"] = tokenized["attention_mask"].to(device)
    batch_of_rows["input_ids"] = tokenized["input_ids"].to(device)
    return batch_of_rows
ds = ds.map(tokenizing_sentences, batched = True, batch_size = PRS["batch_size"])

# Embed the sentences
def embedding_sentences(batch_of_rows : dict):
    batch_of_rows["embedding"] = base_model(
        input_ids = batch_of_rows["input_ids"],
        attention_mask = batch_of_rows["attention_mask"]
    )
    return batch_of_rows

ds = ds.map(tokenizing_sentences, batched = True, batch_size = PRS["batch_size"])