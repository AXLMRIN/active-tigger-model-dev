from toolbox import VisualiseAll


VA = VisualiseAll(filename_baseline = "results/316_results/2025-06-23-Baseline.csv", 
    filename_others = "results/316_results/2025-06-23-Sklearn_classifiers.csv",)

VA.routine(main_title = "316",foldername = "./figures/316_results")