import gc
from torch import Tensor
from torch.cuda import (synchronize, empty_cache, ipc_collect, is_available,
    memory_allocated, memory_reserved)
from transformer_class import dataset, transformer
import numpy as np
from sklearn.metrics import f1_score


if is_available:
        empty_cache()
        ipc_collect()
        synchronize()
gc.collect()

# [5e-6, 1e-5, 2e-5 5e-5]
learning_rate = 1e-4

for learning_rate in [1e-6, 1e-5, 5e-5, 1e-4]:
    ds = dataset("./data/semeval_stance.csv",col_text = "Tweet", col_label = "Stance")
    print(ds)
    tr = None
    try: 
        tr = transformer(ds, "Alibaba-NLP/gte-multilingual-base")

        def preprocess(batch_of_rows : dict):
            """For now we only uncapitalised the sentences"""
            # batch_of_rows["text"] = [sentence.lower() 
            #                             for sentence in batch_of_rows["text"]]
            return batch_of_rows

        tr.preprocess(preprocess)
        tr.encode()

        tr.training_args.disable_tqdm = True
        tr.training_args.output_dir = f"./src3/319_models/2025-06-09-{tr.model_name}-{learning_rate}"
        print(tr.training_args.output_dir)
        tr.encoded_dataset.save_to_disk(f"{tr.training_args.output_dir}-data")

        tr.train()

    except : 
        print("FAILED")

    finally : 
        empty_cache()
        ipc_collect()
        synchronize()

        del tr, ds
        gc.collect()