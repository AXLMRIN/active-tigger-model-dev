from toolbox import RoutineGOfKNN, CustomLogger

logger = CustomLogger("./custom_logs")

LEARNING_RATES = [1e-6, 1e-5, 5e-5, 1e-4]
MODELS = [
    "FacebookAI/roberta-base", 
    "google-bert/bert-base-uncased", 
    # "answerdotai/ModernBERT-base"
]

ranges_of_configs = {
    "learning_rate" : LEARNING_RATES,
    "epoch" : [1,2,3, 4, 5]
}

for model in MODELS:
    routine = RoutineGOfKNN(
        foldername = f"./models/{model}",
        ranges_of_configs = ranges_of_configs,
        n_samples = 1000, 
        logger = logger
    )

    routine.routine("./results/333_results/2025-06-23-Others.csv", n_iterations = 3)

logger.notify_when_done("Optimise KNN done")