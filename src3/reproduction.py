from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from toolbox import routine, cMapper, CustomLogger, routineNotOptmisied

################################################################################
# One layer ML, each model x lr x epoch x  is opitmised 3 times (No optimisation here)
################################################################################

all_models = [
    "src3/2025-05-05-answerdotai/ModernBERT-base",
    "src3/2025-05-05-FacebookAI/roberta-base", 
    "src3/2025-05-05-google-bert/bert-base-uncased"
]

all_lrs = [
    "1e-05",
    "2e-05", 
    "5e-05", 
    "5e-06"
]
# Build cMapper
def layers_mapper_function(value):
    return ()

mapper = cMapper(keys = ["n_neighbors", "metric"],
    functions = [n_neighbors_mapper_function,metric_mapper_function] 
)

# GA parameters 
GA_p = {
    'num_genes' : 1,
    "gene_space" : [
        [0]
    ]
}

# Logger
logger = CustomLogger("src3/pers_logs/MLPOneLayer.txt")

#Loop
routineOneLayer, model, lr = (None,) * 3
for model in all_models:
    for lr in ["1e-05", "2e-05", "5e-05", "5e-06"]:
            for attempt in range(3):

                routineOneLayer = routineNotOptmisied(
                     folder_name = f"{model}-{lr}-data",
                     classifier = MLPClassifier,
                     n_sample_range = [500,1000,1500],
                     epoch_range = [0,1,2,3,4,5],
                     classifier_parameters = {
                          "hidden_layer_sizes" : (),
                          "max_iter" : 1000, 
                          "early_stopping" : True
                        },
                    logger = logger,
                    print_logs = True
                )

                routineOneLayer.run_all()
                routineOneLayer.save_to_csv("src3/results/2025-05-18-MLPOneLayer-2.csv")

CustomLogger().notify_when_done("The MLPOneLayer routine is finished")
del routineOneLayer, model, all_models, lr, all_lrs, GA_p, mapper, logger

################################################################################
# Routine Random Forest, each model x lr x epoch x  is opitmised 3 times
################################################################################

all_models = [
    "src3/2025-05-05-answerdotai/ModernBERT-base",
    "src3/2025-05-05-FacebookAI/roberta-base",
    "src3/2025-05-05-google-bert/bert-base-uncased"
]

all_lrs = [
    "1e-05",
    "2e-05", 
    "5e-05", 
    "5e-06"
]

# Build cMapper
def n_estimators_mapper_function(value):
    return int(value)
def criterion_mapper_function(idx):
    crits = ["gini", "entropy", "log_loss"]
    return crits[int(idx)]
def max_depth_mapper_function(value):
    return int(value)

mapper = cMapper(keys = ["n_estimators", "criterion","max_depth"],
    functions = [n_estimators_mapper_function,criterion_mapper_function,max_depth_mapper_function] 
)

# GA parameters 
GA_p = {
    'num_genes' : 3,
    "gene_space" : [
        {'low' : 10, 'high' : 1000, 'step' : 50},
        [0,1,2],
        [30, 60, 90]
    ]
}

# logger 
logger = CustomLogger("src3/pers_logs/RoutineRandomForest.txt")

#Loop
routineRandomForest, model, lr = (None,) * 3
for model in all_models:
    for lr in all_lrs:
            for attempt in range(3):

                routineRandomForest = routine(
                    folder_name = f"{model}-{lr}-data",
                    classifier = RandomForestClassifier, 
                    n_sample_range = [500,1000,1500],
                    epoch_range = [0,1,2,3,4,5],
                    GA_parameters = GA_p,
                    custom_mapping = mapper,
                    logger = logger,
                    print_logs = True
                )

                routineRandomForest.run_all()
                routineRandomForest.save_to_csv("src3/results/2025-05-18-RandomForest-2.csv")

CustomLogger().notify_when_done("The RandomForest routine is finished")
del routineRandomForest, model, all_models, lr, all_lrs, GA_p, mapper, logger

################################################################################
# Routine KNN, each model x lr x epoch x  is opitmised 3 times
################################################################################

all_models = [
    "src3/2025-05-05-answerdotai/ModernBERT-base",
    "src3/2025-05-05-FacebookAI/roberta-base", 
    "src3/2025-05-05-google-bert/bert-base-uncased"
]

all_lrs = [
    "1e-05",
    "2e-05", 
    "5e-05", 
    "5e-06"
]
# Build cMapper
def n_neighbors_mapper_function(value):
    return int(value)
def metric_mapper_function(idx):
    crits = ["cosine","l1","l2"]
    return crits[int(idx)]

mapper = cMapper(keys = ["n_neighbors", "metric"],
    functions = [n_neighbors_mapper_function,metric_mapper_function] 
)

# GA parameters 
GA_p = {
    'num_genes' : 2,
    "gene_space" : [
        {'low' : 1, 'high' : 20},
        [0,1,2]
    ]
}

# Logger
logger = CustomLogger("src3/pers_logs/RoutineKNN.txt")

#Loop
routineKNN, model, lr = (None,) * 3
for model in all_models:
    for lr in ["1e-05", "2e-05", "5e-05", "5e-06"]:
            for attempt in range(3):
                
                routineKNN = routine(
                    folder_name = f"{model}-{lr}-data",
                    classifier = KNeighborsClassifier, 
                    n_sample_range = [500,1000,1500],
                    epoch_range = [0,1,2,3,4,5],
                    GA_parameters = GA_p,
                    custom_mapping = mapper,
                    logger = logger,
                    print_logs = True 
                )

                routineKNN.run_all()
                routineKNN.save_to_csv("src3/results/2025-05-18-KNN-2.csv")

CustomLogger().notify_when_done("The KNN routine is finished")
del routineKNN, model, all_models, lr, all_lrs, GA_p, mapper, logger