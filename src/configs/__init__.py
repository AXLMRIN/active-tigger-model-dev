c316 = {
    "DEV_MODE" : True, 
    "seed" : 42,
    "filename_open" : "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data/ibc.csv", 
    "filename_open_indexed" : "data/316_ideological_book_corpus/ibc-index",
    "filename_open_embed" : "data/316_ideological_book_corpus/ibc-embedd",
    "csv_train_save" : "models/316_ideological_book_corpus-IdeologySentenceClassifier-train.csv",
    "model" : {
        "name" : "answerdotai/ModernBERT-base",
        "dim" : 768
    },
    "batch_size" : 32,
    "n_epoch" : 40,
    "tokenizing" : {
        "padding" : "max_length",
        "truncation" : True,
        "max_length" : 42,
        "return_tensors" : "pt"
    },
    "DataLoader" : {
        "num_workers" : 0,
        "pin_memory" : True
    },
    "optimizer" : {
        "lr" : 1e-3
    }
}