from toolbox import (
    AutoClassifierRoutineConfig, AutoClassifierRoutine
)
from mergedeep import merge

general_args = {
    "sentence_col" : "sentence",
    "label_col" : "leaning",
    "batch_size" : 16,
    "num_train_epochs" : 10,
    "only_train_classifier" : False,
    "dev_mode" : True
}

file_args = {
    "316" : {"files":
        {
            "open_local" : "data/316_ideological_book_corpus/ibc.csv",
            "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data.csv", 
            "output_dir" : "2025-03-17-autoClassifier-test"
        }
    }
}

for test_id in file_args : 
    args = merge(general_args, file_args[test_id])
    config = AutoClassifierRoutineConfig(**args)
    routine = AutoClassifierRoutine(config)
    routine.run()