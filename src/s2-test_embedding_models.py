from toolbox import TestAllEpochs, CustomLogger

logger = CustomLogger("./custom_logs")

TestAllEpochs("./models/FacebookAI/roberta-base/001", logger = logger).\
    routine("2025-06-14-TEST.csv", 
            additional_tags = {"classifier" : "Baseline - HF Classifier"})