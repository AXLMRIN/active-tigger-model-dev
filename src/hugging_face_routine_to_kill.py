from transformer_class import transformer, dataset
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from torch import Tensor
import numpy as np
from tqdm import tqdm
import pandas as pd
from toolbox import CustomLogger

root = "./src3"
device = "cpu"

all_models = [
    "answerdotai/ModernBERT-base",
    "FacebookAI/roberta-base",
    "google-bert/bert-base-uncased"
]

save = []
try : 
    for model in tqdm(all_models, position = 0, leave = True):
        for lr in ["1e-05", "2e-05", "5e-05", "5e-06"]:
            ds = load_from_disk(f"{root}/2025-05-05-{model}-{lr}-data/test")
            for i, checkpoint in enumerate(["19","38","57","76","95"]): 
                inference_model = AutoModelForSequenceClassification.\
                            from_pretrained(f"{root}/2025-05-05-{model}-{lr}/checkpoint-{checkpoint}").\
                            to(device = device)

                labels_true = []
                labels_pred = []
                for batch in ds.batch(16):
                    result = inference_model(**{
                        'input_ids' : Tensor(batch['input_ids']).to(dtype = int).to(device=device), 
                        'attention_mask' : Tensor(batch['attention_mask']).to(dtype = int).to(device=device) 
                    }).logits.detach().numpy()
                    labels_true.extend([np.argmax(row).item() for row in batch["labels"]])
                    labels_pred.extend([np.argmax(row).item() for row in result])

                score = f1_score(labels_true, labels_pred, average='macro')
                save.append({
                    'model' : model,
                    'lr' : lr,
                    'epoch' : i+1,
                    'f1_macro' : score
                })




    pd.DataFrame(save).to_csv(f"{root}/results/HuggingFaceClassification.csv", index = False)
    CustomLogger().notify_when_done("loop huggingface")
except : 
    print(save)
    CustomLogger().notify_when_done("FAIL loop huggingface")