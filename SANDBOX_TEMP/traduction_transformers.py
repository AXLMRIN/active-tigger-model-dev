from transformer_class import dataset, transformer, CustomLogger
from datasets.utils.logging import disable_progress_bar
from torch import Tensor
from torch.utils.data import DataLoader
from torch.cuda import synchronize, ipc_collect, empty_cache
from transformer_functions import multi_label_metrics
import pygad 
import numpy as np

import gc

disable_progress_bar()

# Solution
# 0 - learning rate
# 1 - weight decay

def fitness_func(ga_instance, solution, solution_idx):
    ds = dataset("dataUNSAFE/ibc.csv",
                col_text = "sentence", 
                col_label = "leaning")

    tr = transformer(ds, "google-bert/bert-base-uncased")

    def preprocess(batch_of_rows : dict):
        """For now we only uncapitalised the sentences"""
        batch_of_rows["text"] = [sentence.lower() 
                                    for sentence in batch_of_rows["text"]]
        return batch_of_rows

    tr.preprocess(preprocess)
    tr.encode()

    tr.training_argslearning_rate = solution[0]
    tr.training_args.weight_decay = solution[1]
    
    tr.train()
    print(np.array(tr.encoded_dataset["test"]["input_ids"]).shape)
    test_loader = DataLoader(tr.encoded_dataset["test"], 
                    batch_size = tr.training_args.per_device_train_batch_size,
                    shuffle = True)
    logits = []
    labels = []
    for batch in test_loader: 
        output = tr.model(**{
            "input_ids" : Tensor([t.tolist() for t in batch["input_ids"]]).T.\
                squeeze().int().to(device=tr.model.device),
            "attention_mask" : Tensor([t.tolist() for t in batch["attention_mask"]]).T.\
                squeeze().to(device=tr.model.device)
        })

        logits.extend(output.logits.tolist())
        labels.extend(Tensor([t.tolist() for t in batch["labels"]]).T.tolist())

    print(Tensor(logits).shape)
    print(Tensor(labels).shape)
    result = multi_label_metrics(
            Tensor(logits).to(device="cpu"),
            Tensor(labels).to(device="cpu")
            )['f1']
    print(result)
    
    if tr.device == "cuda":
        synchronize()
        empty_cache()
        ipc_collect()
    del tr, ds
    gc.collect()
    return result

GA_parameters = {
    #Must Specify
    'fitness_func' : fitness_func,
    'num_generations' : 50,
    
    'sol_per_pop' : 10,
    'num_parents_mating' : 2,
    'keep_elitism' : 1,
    
    'num_genes' : 2,
    "gene_space" : [
        {'low' : 1e-6, 'high' : 1e-3},
        [0.1,0.2,0.3,0.4]
    ],
    "stop_criteria" : "saturate_5",
    # Default
    'mutation_type' : "random",
    'parent_selection_type' : "sss",
    'crossover_type' : "single_point",
    'mutation_percent_genes' : 50,
    # Other
    'save_solutions' : False,
    'random_seed' : 2306406
}

ga_instance = pygad.GA(**GA_parameters)

try : 
    ga_instance.run()
    ga_instance.save("2025-04-28-fine_tuning")
    
    CustomLogger().notify_when_done(f"Run finished\n{ga_instance.best_solution()}")

except:
    CustomLogger().notify_when_done("Run failed")

