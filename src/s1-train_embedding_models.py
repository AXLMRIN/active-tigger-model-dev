from toolbox import DataHandler, CustomTransformersPipeline, clean, CustomLogger

DH, pipe = None, None
logger = CustomLogger("./custom_logs")
try : 
    DH = DataHandler(
        filename = "./data/316_ideological_book_corpus/ibc.csv",
        text_column = "sentence", 
        label_column = "leaning",
        logger = logger
    )
    DH.routine(stratify_columns="LABEL")
    pipe = CustomTransformersPipeline(data = DH, 
        model_name = "FacebookAI/roberta-base", 
        logger = logger)
    pipe.routine(debug_mode=True)

except Exception as error: 
    print("### ERROR " + "#" * 40)
    print(error)
    print("#" * 50)

finally:
    del DH, pipe
    clean()