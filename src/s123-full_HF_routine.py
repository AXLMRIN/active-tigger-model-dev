from toolbox import (
    DataHandler, 
    CustomTransformersPipeline, 
    TestAllEpochs,
    ExportEmbeddingsForAllEpochs,
    clean, 
    CustomLogger)

logger = CustomLogger("./custom_logs")

LEARNING_RATES = [1e-6, 1e-5, 5e-5, 1e-4]
MODELS = [
    "FacebookAI/roberta-base", 
    "google-bert/bert-base-uncased", 
    "answerdotai/ModernBERT-base"
]


for model in MODELS :
    for learning_rate in LEARNING_RATES:
        DH, pipe = None, None
        try : 
            # Step 1 - Training
            DH = DataHandler(
                filename = "./data/media_ideology_split.csv",
                text_column = "content", 
                label_column = "bias_text",
                logger = logger,
                tokenizer_max_length = 1e5, # Absurdly high so that the number of max_tokenizer is set to the model's limit
                disable_tqdm = True
            )
            DH.routine(stratify_columns="LABEL")

            pipe = CustomTransformersPipeline(
                data             = DH, 
                model_name       = model,
                learning_rate    = learning_rate,
                num_train_epochs = 5,
                logger           = logger
            )
            pipe.routine(debug_mode = False)

            output_dir = pipe.output_dir
            del DH, pipe
            clean()
            DH, pipe = (None, None)

            # Step 2 - Testing
            TestAllEpochs(output_dir, logger = logger).\
                routine(
                    "./results/333_results/HuggingFace_Baseline.csv", 
                    additional_tags = {"classifier" : "Baseline - HF Classifier"}
                )

            # Step 3 - Saving embeddings
            ExportEmbeddingsForAllEpochs(output_dir, logger = logger).\
                routine(delete_files_after_routine=True)
        
        except Exception as e:
            print("#" * 100)
            print(f"ERROR during {model} - {learning_rate}")
            print(e)
            print("#" * 100)

        finally : 
            del DH, pipe
            clean()

logger.notify_when_done("Routine S123 finished")