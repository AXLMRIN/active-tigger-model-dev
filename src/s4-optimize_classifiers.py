from toolbox import RoutineGOfSC
from sklearn.neighbors import KNeighborsClassifier

ranges_of_configs = {
    "learning_rate" : [5e-5, 5e-4, 2e-5],
    "weight_decay" : [0.05, 0.01], 
    "epoch" : [1,2]
}

def n_neighbors_mapper_function(value):
    return int(value)
def metric_mapper_function(idx):
    crits = ["cosine","l1","l2"]
    return crits[int(idx)]

routine = RoutineGOfSC(
    foldername = "FacebookAI/roberta-base",
    ranges_of_configs = ranges_of_configs,
    classifier = KNeighborsClassifier, 
    parameters_mapper = {
        "n_neighbors" : n_neighbors_mapper_function,
        "metric" : metric_mapper_function
    },
    gene_space = {
        'num_genes' : 2,
        "gene_space" : [
            {'low' : 1, 'high' : 20},
            [0,1,2]
        ]
    },
    n_samples = 500
)

routine.routine("2025-06-16-TEST.csv", n_iterations = 2)