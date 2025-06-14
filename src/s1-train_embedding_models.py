from toolbox import DataHandler, CustomTransformersPipeline, clean

try : 
    DH = DataHandler(
        filename = "316_ideological_book_corpus/ibc.csv",
        text_column = "sentence", 
        label_column = "leaning"
    )
    DH.routine(stratify_columns="LABEL")
    pipe = CustomTransformersPipeline(data = DH, 
        model_name = "FacebookAI/roberta-base")
    pipe.routine(debug_mode=True)

except Exception as error: 
    print("### ERROR " + "#" * 40)
    print(error)
    print("#" * 50)

finally:
    del DH, pipe
    clean()