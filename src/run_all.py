from toolbox import (
    AutoClassifierRoutineConfig, AutoClassifierRoutine
)

args = {
    "files" : {
        "open_local" : "data/316_ideological_book_corpus/ibc.csv",
        "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data/ibc.csv", 
        "output_dir" : "2025-03-17-autoClassifier-test"
    },
    "sentence_col" : "sentence",
    "label_col" : "leaning",
    "batch_size" : 16,
    "num_train_epochs" : 10,
    "only_train_classifier" : False,
    "dev_mode" : True
}

config = AutoClassifierRoutineConfig(**args)
routine = AutoClassifierRoutine(config)
routine.run()