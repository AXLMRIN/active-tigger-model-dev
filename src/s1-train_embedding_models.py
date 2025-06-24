from toolbox import DataHandler, CustomTransformersPipeline, clean, CustomLogger

DH, pipe = None, None
logger = CustomLogger("./custom_logs")

LEARNING_RATES = [1e-6, 1e-5, 5e-5, 1e-4]
MODELS = [
    "FacebookAI/roberta-base", 
    "google-bert/bert-base-uncased", 
    "answerdotai/ModernBERT-base"
]
for learning_rate in LEARNING_RATES:
    for model in MODELS :
        try : 
            DH = DataHandler(
                filename = "./data/media_ideology_split.csv",
                text_column = "content", 
                label_column = "bias_text",
                logger = logger
            )
            DH.routine(stratify_columns="LABEL")
            pipe = CustomTransformersPipeline(
                data = DH, 
                model_name = model,
                learning_rate = learning_rate,
                logger = logger)
            pipe.routine(debug_mode = False)

        except Exception as error: 
            print("### ERROR " + "#" * 40)
            print(error)
            print("#" * 50)

        finally:
            del DH, pipe
            clean()