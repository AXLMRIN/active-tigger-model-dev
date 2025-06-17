from toolbox import TestAllEpochs


TestAllEpochs("./models/google-bert/bert-base-uncased/001").\
    routine("2025-06-14-TEST.csv", 
            additional_tags = {"classifier" : "Baseline - HF Classifier"})