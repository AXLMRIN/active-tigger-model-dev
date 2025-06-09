from datasets import load_from_disk
import gc
import os
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from torch import Tensor
from torch.cuda import (synchronize, empty_cache, ipc_collect)
import numpy as np
import pandas as pd

model_name = "answerdotai/ModernBERT-base"

root_file = "./src3/319_models/2025-06-09"

save = []

for learning_rate in ["1e-06", "5e-05","1e-05", "0.0001"]:
    checkpoints = os.listdir(
        f"{root_file}-{model_name}-{learning_rate}/")
    checkpoints = sorted(checkpoints, key = lambda x : int(x.split("-")[-1]))
    print(checkpoints)
    for epoch, checkpoint in enumerate(checkpoints):
        
        ds, model = None, None
        
        try: 
            ds = load_from_disk(
                f"{root_file}-{model_name}-{learning_rate}-data")

            model = AutoModelForSequenceClassification.from_pretrained(
                f"{root_file}-{model_name}-{learning_rate}/{checkpoint}").\
                to(device="cuda")

            labels_true = []
            labels_pred = []
            for batch in ds["test"].batch(16):
                result = model(
                    **{
                        'input_ids' : Tensor(batch['input_ids']).\
                                        to(dtype = int).to(device="cuda"), 
                        'attention_mask' : Tensor(batch['attention_mask']).\
                                        to(dtype = int).to(device="cuda") 
                    }).\
                    logits.detach().cpu().numpy()
                labels_true.extend([np.argmax(row).item() for row in batch["labels"]])
                labels_pred.extend([np.argmax(row).item() for row in result])
                break
            score = f1_score(labels_true, labels_pred, average='macro')

            save.append({
                "model" : model_name,
                "lr" : float(learning_rate),
                "epoch" : epoch + 1, 
                "f1_macro" : score,
                "filename" : f"{root_file}-{model_name}-{learning_rate}-data",
                "n_samples" : "Entraînement Hugging Face",
                "iteration" : 1,
                "time" : np.nan
            })

        except Exception as e: 
            print(f"FAILED : {e}")
        finally : 
            empty_cache()
            ipc_collect()
            synchronize()

            del model, ds
            gc.collect()
        break
    break

pd.DataFrame(save).to_csv("./src3/results/2025-06-09-HuggingFaceClassification.csv", index = False)