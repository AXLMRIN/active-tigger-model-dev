 
from datasets import Dataset, DatasetDict
import pandas as pd
from random import shuffle

model_name = "google-bert/bert-base-uncased"
# Load and shuffle
df = pd.read_csv("../../dataUNSAFE/ibc.csv").sample(frac = 1)

# Group per label
grouped = df.groupby("leaning")
print(grouped.size())

# Number of element to keep per label
N_train = 400 
N_eval = 100
N_test = 100

LABEL2ID = {}
ID2LABEL = {}
LABEL = []
for id, (label, _) in enumerate(grouped):
    LABEL2ID[label] = id
    ID2LABEL[id] = label
    LABEL.append(label)

N_LABEL = len(LABEL2ID)

def concat_shuffle(start,finish):
    return pd.concat([grouped.get_group(label)[start:finish]
                        for label in LABEL2ID]).\
                sample(frac = 1)


dataset = DatasetDict({
    "train" : Dataset.from_pandas(concat_shuffle(0,N_train)),
    "eval" : Dataset.from_pandas(concat_shuffle(N_train, N_train + N_eval)),
    "test" : Dataset.from_pandas(concat_shuffle(N_train + N_eval,N_train + N_eval + N_test))
})

def preprocess(batch_of_rows : dict):
    """For now we only uncapitalised the sentences"""
    batch_of_rows["sentence"] = [sentence.lower() 
                                 for sentence in batch_of_rows["sentence"]]
    batch_of_rows["leaning"] = [LABEL2ID[leaning] 
                                for leaning in batch_of_rows["leaning"]]
    for label in LABEL2ID:
        batch_of_rows[label] = [LABEL2ID[label] == leaning_id
                                for leaning_id in batch_of_rows["leaning"]]
    return batch_of_rows

dataset["train"] = dataset["train"].map(preprocess, batched = True, batch_size = 64)
dataset["eval"] = dataset["eval"].map(preprocess, batched = True, batch_size = 64)
dataset["test"] = dataset["test"].map(preprocess, batched = True, batch_size = 64)

 
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer
import numpy as np 

def preprocess_data(batch_of_rows : LazyBatch, tokenizer : BertTokenizerFast,
         labels : list[int], label2id : dict[str:int],
         sentence_column : str = "Tweet",) -> BatchEncoding:
    # Takes in a batch of rows (as a : LazyBatch ~ dataframe ish) 
    
    # collect the text and tokenize it 
    text = batch_of_rows[sentence_column]
    encoding : BatchEncoding = tokenizer(
        text, padding = "max_length", truncation = True, max_length = 128 
    )
    # Create a mattrix collecting all the metadata (emotions associated to the 
    # tweet)
    labels_matrix = np.zeros((len(text), len(labels)))
    for label in label2id:
        labels_matrix[:,label2id[label]] = batch_of_rows[label]

    # Associate the metadata to the encodings
    encoding["labels"] = labels_matrix.tolist()
    return encoding

tokenizer = AutoTokenizer.from_pretrained(model_name)

encoded_dataset = dataset.map(
    lambda batch_of_rows : preprocess_data(batch_of_rows,tokenizer, LABEL, LABEL2ID,
        sentence_column = "sentence"), 
    batched = True, remove_columns=dataset["train"].column_names
)

 
from transformers import AutoModelForSequenceClassification, AutoConfig

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type = "multi_label_classification", 
    num_labels = N_LABEL,
    id2label = ID2LABEL, label2id = LABEL2ID
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)

 
from transformers import TrainingArguments, Trainer


total_per_batch = 64
batch_size_device = 8
metric_name = "f1"
training_args = TrainingArguments(
    num_train_epochs=10,
    bf16=True,
    # Hyperparameters
    learning_rate=2e-5,
    # optim_args = {}
    weight_decay=0.01,
    warmup_ratio = 0.1,
    # Second order hyperparameters
    per_device_train_batch_size = batch_size_device,
    per_device_eval_batch_size = batch_size_device,
    gradient_accumulation_steps = int(total_per_batch/ batch_size_device),
    optim = "adamw_torch",
    # Metrics
    metric_for_best_model=metric_name,
    # Pipe
    output_dir = "2025-04-23-bert-GA",
    overwrite_output_dir=True,
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    torch_empty_cache_steps = int(len(dataset["train"]) / batch_size_device),
    load_best_model_at_end=True,
    save_total_limit = 1,

    # disable_tqdm = True
)
    

 
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn import Sigmoid
from transformers import EvalPrediction

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/ 
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

 
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["eval"],
    compute_metrics = compute_metrics,
    # optimizers = 
)

 
trainer.train()

 



