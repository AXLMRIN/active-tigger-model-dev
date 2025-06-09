from datasets import load_from_disk
import gc
import os
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from torch import Tensor
from torch.cuda import (synchronize, empty_cache, ipc_collect)
import numpy as np
import pandas as pd

model_name = "FacebookAI/roberta-base"

root_file = "./src3/319_models/2025-06-09"

save = []

for learning_rate in ["1e-06", "5e-05","1e-05", "0.0001"]:
    # Get all the checkpoints
    checkpoints = os.listdir(
        f"{root_file}-{model_name}-{learning_rate}/")
    checkpoints = sorted(checkpoints, key = lambda x : int(x.split("-")[-1]))

    # Save the epoch 0
    save.append({
        "model" : model_name,
        "lr" : float(learning_rate),
        "epoch" : 0, 
        "f1_macro" : 0.33, # Random classifier for 3 classes
        "filename" : f"{root_file}-{model_name}-{learning_rate}-data",
        "n_samples" : "Entraînement Hugging Face",
        "iteration" : 1,
        "time" : np.nan
    })

    # Run for the 5 other epochs
    for epoch, checkpoint in enumerate(checkpoints):
        
        ds, model = None, None
        
        try: 
            # Load the data
            ds = load_from_disk(
                f"{root_file}-{model_name}-{learning_rate}-data")

            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(
                f"{root_file}-{model_name}-{learning_rate}/{checkpoint}").\
                to(device="cuda")

            # Testing loop
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

            # Evaluate the score
            score = f1_score(labels_true, labels_pred, average='macro')

            # Save the score
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
df = pd.read_csv("./src3/results/2025-06-09-HuggingFaceClassification.csv")
df = pd.concat((df, pd.DataFrame(save)))
df.to_csv("./src3/results/2025-06-09-HuggingFaceClassification.csv", index = False)