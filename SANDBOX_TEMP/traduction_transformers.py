from transformer_class import dataset, transformer
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

ds = dataset("data/316_ideological_book_corpus/ibc.csv",
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
print(tr.train())