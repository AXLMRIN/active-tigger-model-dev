from toolbox import RoutineGOfKNN, CustomLogger

logger = CustomLogger("./custom_logs")

ranges_of_configs = {
    "learning_rate" : [1e-5,5e-5,1e-4],
    "epoch" : [1,2,3]
}

routine = RoutineGOfKNN(
    foldername = "./models/FacebookAI/roberta-base",
    ranges_of_configs = ranges_of_configs,
    n_samples = 500, 
    logger = logger
)

routine.routine("2025-06-17-TEST-RoutineKNN.csv", n_iterations = 2)