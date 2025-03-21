from toolbox import (
    AutoClassifierRoutineConfig, AutoClassifierRoutine
)
from mergedeep import merge
from copy import deepcopy
from logging import getLogger, INFO, basicConfig
from datetime import datetime
import gc

general_args = {
    "batch_size" : 64,
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
            "output_dir" : (f"saves_model/{datetime.today().strftime('%d-%m-%Y')}"
                            "-autoClassifier-test-316")
        },
        "tokenizer_max_length" : 100
    },
    "319" : {
        "sentence_col" : "Tweet",
        "label_col" : "Stance",
        "files" : {
            "open_local" : "data/319_semeval_stance/semeval_stance.csv",
            "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/319_stance/data.csv", 
            "output_dir" : (f"saves_model/{datetime.today().strftime('%Y-%m-%d')}"
                            "-autoClassifier-test-319")
        },
        "tokenizer_max_length" : 60
    },
    # FIXME Not running
    # "333" : {
    #     "batch_size" : 8,
    #     "sentence_col" : "content",
    #     "label_col" : "bias_text",
    #     "files" : {
    #         "open_local" : "data/333_media_ideology/media_ideology.csv",
    #         "open_s3" : "s3://projet-datalab-axel-morin/model_benchmarking/333_ideology/data.csv", 
    #         "output_dir" : "2025-03-17-autoClassifier-test-333"
    #     },
    #     "tokenizer_max_length" : 2745
    # },
    # FIXME might not be used 
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
logger = getLogger(f"GENERAL_LOGGER - {datetime.today().strftime('%d-%m-%Y; %H:%M')}")
basicConfig(filename='general_logger.log', encoding='utf-8', level= INFO)
logger.info("PROCESS STARTED")
for test_id in file_args : 
    logger.info(f"Running test number {test_id} =================")
    args = merge(deepcopy(general_args), file_args[test_id])
    config = AutoClassifierRoutineConfig(**args)
    routine = AutoClassifierRoutine(config)
    routine.run()

    del config, routine
    gc.collect()