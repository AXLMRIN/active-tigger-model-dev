from transformer_class import dataset, transformer
from datasets.utils.logging import disable_progress_bar
from torch import Tensor
from transformer_functions import multi_label_metrics

disable_progress_bar()

def fitness(ga_instance, solution, solution_idx):
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
    tr.debug_mode()
    tr.train()

    output = tr.model(**{
        "input_ids" : Tensor(tr.encoded_dataset["eval"]["input_ids"]).squeeze().int(),
        "attention_mask" : Tensor(tr.encoded_dataset["eval"]["attention_mask"]).squeeze()
    })

    return multi_label_metrics(output.logits,tr.encoded_dataset["eval"]["labels"])['f1']

fitness(None,None, None)