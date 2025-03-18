from toolbox import (
    AutoClassifierRoutineConfig, AutoClassifierRoutine
)
from mergedeep import merge

general_args = {
    "batch_size" : 16,
    "num_train_epochs" : 10,
    "only_train_classifier" : False,
    "dev_mode" : True
}

file_args = {
    "316" : {
        "sentence_col" : "sentence",
        "label_col" : "leaning",
        "files" : {
            "open_local" : "data/316_ideological_book_corpus/ibc.csv",
            "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/316_ideology/data.csv", 
            "output_dir" : "2025-03-17-autoClassifier-test-316"
        }
    },
    "319" : {
        "sentence_col" : "Tweet",
        "label_col" : "Stance",
        "files" : {
            "open_local" : "data/319_semeval_stance/semeval_stance.csv",
            "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/319_stance/data.csv", 
            "output_dir" : "2025-03-17-autoClassifier-test-319"
        }
    },
    "333" : {
        "sentence_col" : "content",
        "label_col" : "bias_text",
        "files" : {
            "open_local" : "data/333_media_ideology/media_ideology.csv",
            "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/333_ideology/data.csv", 
            "output_dir" : "2025-03-17-autoClassifier-test-333"
        }
    },
    # FIXME the preprocess is stuck
    # "334" : {
    #     "sentence_col" : "Quotes",
    #     "label_col" : "TropesHuman",
    #     "files" : {
    #         "open_local" : "data/334_tropes/tropes.csv",
    #         "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/334_roles/data.csv", 
    #         "output_dir" : "2025-03-17-autoClassifier-test-334"
    #     }
    # }
}

for test_id in file_args : 
    print(("\n"
        f"Running test number {test_id} =================\n\n"
    ))

    args = merge(general_args, file_args[test_id])
    config = AutoClassifierRoutineConfig(**args)
    routine = AutoClassifierRoutine(config)
    routine.run()