from toolbox import VisualiseAll


VA = VisualiseAll(filename_baseline = "./results/debug/2025-06-14-TEST.csv", 
    filename_others = "./results/debug/2025-06-17-TEST-RoutineKNN.csv",)

VA.routine(foldername = "./figures/debug")