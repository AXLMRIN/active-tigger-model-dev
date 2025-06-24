from toolbox import TestAllEpochs, CustomLogger

logger = CustomLogger("./custom_logs")

MODELS = [
    "FacebookAI/roberta-base", 
    "google-bert/bert-base-uncased", 
    "answerdotai/ModernBERT-base"
]

for model in MODELS :
    for i in range(1,5):
        TestAllEpochs(f"./models/{model}/00{i}", logger = logger).\
            routine(
                "./results/333_results/HuggingFace_Baseline.csv", 
                additional_tags = {"classifier" : "Baseline - HF Classifier"}
            )