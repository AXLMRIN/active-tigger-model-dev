# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import load_dataset, DatasetDict
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
from toolbox.IdeologySentenceClassifier import IdeologySentenceClassifier

# PARAMETERS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
FILENAME : str = "data/316_ideological_book_corpus/ibc.csv"
train_record_save_filename : str = "316_ideological_book_corpus-IdeologySentenceClassifier-train.csv"
