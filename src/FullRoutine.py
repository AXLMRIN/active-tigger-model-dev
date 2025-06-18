# IMPORTS ######################################################################
import os
from toolbox.train_embedding_models import DataHandler, CustomTransformersPipeline
from toolbox.test_embedding_models import TestAllEpochs
from toolbox.save_embeddings import ExportEmbeddingsForAllEpochs
from toolbox.optimize_classifiers import RoutineGOfKNN
from toolbox import clean

# PARAMETERS ###################################################################
DEBUG_MODE = True

data_input_filename = "./data/316_ideological_book_corpus/ibc.csv"
data_input_text_column = "sentence"
data_input_label_column = "leaning"

pipeline_training_model_names = ["FacebookAI/roberta-base"]
pipeline_training_learning_rates = [5e-6, 1e-5, 5e-5, 1e-4]
pipeline_training_weight_decay = [0.005, 0.01, 0.05]
pipeline_training_num_train_epochs = 3

testing_embedding_models_output_file = ("./results/"
    "2025-06-18-HuggingFace-embedding-results.csv")
testing_embedding_models_additional_tags = {
    "classifier" : "Baseline - HF Classifier"
}

embedding_saver_delete_files_after_routine = False # TODO Change

optimising_classifiers_ranges_of_configs = {
    "learning_rate" : pipeline_training_learning_rates,
    "weight_decay" : pipeline_training_weight_decay,
    "epoch" : [e for e in range(1,pipeline_training_num_train_epochs + 1)]
}
optimising_classifiers_output_file = "./results/2025-06-19-Classifiers_optimised.csv"
optimising_classifiers_n_iterations = 2
optimising_classifiers_n_samples = [50,100]
# ROUTINE ######################################################################
for model_name in pipeline_training_model_names:
    for learning_rate in pipeline_training_learning_rates:
        for weight_decay in pipeline_training_weight_decay:
            data = DataHandler(
                filename      = data_input_filename,
                text_column  = data_input_text_column,
                label_column = data_input_label_column
            )
            data.routine(stratify_columns="LABEL")
            pipe = CustomTransformersPipeline(
                data             = data,
                model_name       = model_name,
                learning_rate    = learning_rate,
                weight_decay     = weight_decay,
                num_train_epochs = pipeline_training_num_train_epochs
            )
            pipe.routine(DEBUG_MODE)
            # Clean the loop
            del pipe, data
            clean()
print("-- Step 1 : Done --")

for model_name in pipeline_training_model_names:
    for iteration in os.listdir(f"./models/{model_name}/"):
        testingRoutine = TestAllEpochs(f"./models/{model_name}/{iteration}")
        testingRoutine.routine(
            filename         = testing_embedding_models_output_file,
            additional_tags = testing_embedding_models_additional_tags
        )
        del testingRoutine
        clean()
print("-- Step 2 : Done --")

for model_name in pipeline_training_model_names:
    for iteration in os.listdir(f"./models/{model_name}/"):
        embedding_saver = ExportEmbeddingsForAllEpochs(f"./models/{model_name}/{iteration}")
        embedding_saver.routine(
            delete_files_after_routine=embedding_saver_delete_files_after_routine
        )
        del embedding_saver
        clean()
print("-- Step 3 : Done --")

for model_name in pipeline_training_model_names:
    for n_samples in optimising_classifiers_n_samples:
        optimising_classifiers = RoutineGOfKNN(
            foldername       = f"./models/{model_name}",
            ranges_of_configs = optimising_classifiers_ranges_of_configs,
            n_samples        = n_samples
        )
        optimising_classifiers.routine(
            filename      = optimising_classifiers_output_file, 
            n_iterations = optimising_classifiers_n_iterations,
        )
        del optimising_classifiers
        clean()
print("-- Step 4 : Done --")