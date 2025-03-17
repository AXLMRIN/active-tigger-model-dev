from datasets import Dataset
from pandas import read_csv
import torch
from torch import float32
from torch.cuda import is_available as gpu_available
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, 
      TrainingArguments, Trainer
)

from toolbox import storage_options, split_test_train_valid
from configs import c316; PRS = c316
from toolbox.bert_train_tutorial import (
    preprocess_data, compute_metrics
)
import os
# PARAMETERS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
float_dtype = float32
# Static parameters

from configs import c316; PRS = c316

# Dynamic parameters
att_implementation : str = "sdpa"
# TODO Demander pour flash_attention_2
device = "cuda" if gpu_available() else "cpu"
print(f"Running on {device}.")

if device == "cuda" : torch.set_float32_matmul_precision('high')
# SCRIPT --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
try :
    ds : Dataset = Dataset.from_pandas(read_csv(
        "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data/ibc.csv", 
        storage_options = {
            'client_kwargs': {'endpoint_url': 'https://minio-simple.lab.groupe-genes.fr'},
            'key': os.environ["AWS_ACCESS_KEY_ID"],
            'secret': os.environ["AWS_SECRET_ACCESS_KEY"],
            'token': os.environ["AWS_SESSION_TOKEN"]
        }
    ))
    print("ds loaded with s3")
except:
    ds : Dataset = Dataset.from_pandas(read_csv(
        "data/316_ideological_book_corpus/ibc.csv"
    ))
    print("ds loaded on local")

LABEL : list[str] = list(set(ds["leaning"])); n_labels : int = len(LABEL)
ID2LABEL : dict[int:str] = {i : cat for i,cat in enumerate(LABEL)}
LABEL2ID : dict[str:int] = {cat:i for i,cat in enumerate(LABEL)}
print("Categories : " + ", ".join([cat for cat in LABEL]),"\n")
ds = split_test_train_valid(ds)
# Preprocess - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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

ds = ds.map(preprocess, batched = True, batch_size = PRS["batch_size"])
print(">>> Preprocess - Done")
# Tokenize - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

encoded_dataset = ds.map(
    lambda batch_of_rows : preprocess_data(batch_of_rows,tokenizer, LABEL, LABEL2ID,
        sentence_column = "sentence"), 
    batched = True, remove_columns=ds["train"].column_names
)
# Loading the model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    problem_type = "multi_label_classification", num_labels = len(LABEL),
    id2label = ID2LABEL, label2id = LABEL2ID
).to(device)

for name, param in model.named_parameters():
    if name.startswith("classifier") : param.requires_grad = True
    else : param.requires_grad = False

# Train the model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
batch_size = 8
metric_name = "f1"
training_args = TrainingArguments(
    output_dir = "2025-03-17-autoClassifier-test",
    overwrite_output_dir=True,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_steps=0.25,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

# For debug purposes
encoded_dataset["train"] = encoded_dataset["train"].select(range(0,20))
encoded_dataset["validation"] = encoded_dataset["validation"].select(range(0,20))
#---

trainer = Trainer(model, training_args,
                  train_dataset = encoded_dataset["train"],
                  eval_dataset = encoded_dataset["validation"],
                  processing_class = tokenizer,
                  compute_metrics = compute_metrics
)


print(">>> Start training")
# trainer.train()
print(">>> Training - Done")
