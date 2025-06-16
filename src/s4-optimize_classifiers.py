from toolbox import RoutineGOfKNN

ranges_of_configs = {
    "learning_rate" : [5e-5, 5e-4, 2e-5],
    "weight_decay" : [0.05, 0.01], 
    "epoch" : [1,2]
}

routine = RoutineGOfKNN(
    foldername = "FacebookAI/roberta-base",
    ranges_of_configs = ranges_of_configs,
    n_samples = 500
)

routine.routine("2025-06-16-TEST-RoutineKNN.csv", n_iterations = 2)