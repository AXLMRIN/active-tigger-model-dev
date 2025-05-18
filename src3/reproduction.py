from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from toolbox import routine, cMapper, CustomLogger

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

# Routine Random Forest, each model x lr x epoch x  is opitmised 3 times

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
for model in all_models:
    for lr in all_lrs:
            for attempt in range(3):

                routineRandomForest = routine(
                    folder_name = f"{model}-{lr}-data",
                    classifier = RandomForestClassifier, 
                    n_sample_range = [250,500,750,1000,1500],
                    epoch_range = [0,1,2,3,4,5],
                    GA_parameters = GA_p,
                    custom_mapping = mapper,
                    logger = logger,
                    print_logs = False
                )

                routineRandomForest.run_all()
                routineRandomForest.save_to_csv("src3/results/2025-05-18-RandomForest-2.csv")

CustomLogger().notify_when_done("The RandomForest routine is finished")
del routineRandomForest, model, lr, GA_p, mapper, logger

# Routine KNN, each model x lr x epoch x  is opitmised 3 times

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
for model in all_models:
    for lr in ["1e-05", "2e-05", "5e-05", "5e-06"]:
            for attempt in range(3):

                routineKNN = routine(
                    folder_name = f"{model}-{lr}-data",
                    classifier = KNeighborsClassifier, 
                    n_sample_range = [250,500,750,1000,1500],
                    epoch_range = [0,1,2,3,4,5],
                    GA_parameters = GA_p,
                    custom_mapping = mapper,
                    logger = logger,
                    print_logs = False 
                )

                routineKNN.run_all()
                routineKNN.save_to_csv("src3/results/2025-05-18-KNN-2.csv")

CustomLogger().notify_when_done("The RandomForest routine is finished")


# def basicML(d: DATA):
#     clf = MLPClassifier(hidden_layer_sizes=(),max_iter = 1000, early_stopping=True)
#     clf.fit(d.X_train, d.y_train)
#     return f1_score(y_true=d.y_test, y_pred=clf.predict(d.X_test), average='macro')

# save = []
# for folder_name in [
#         "2025-05-05-answerdotai/ModernBERT-base-1e-05-data",
#         "2025-05-05-answerdotai/ModernBERT-base-2e-05-data",
#         "2025-05-05-answerdotai/ModernBERT-base-5e-05-data",
#         "2025-05-05-answerdotai/ModernBERT-base-5e-06-data",
#         "2025-05-05-FacebookAI/roberta-base-1e-05-data",
#         "2025-05-05-FacebookAI/roberta-base-2e-05-data",
#         "2025-05-05-FacebookAI/roberta-base-5e-05-data",
#         "2025-05-05-FacebookAI/roberta-base-5e-06-data",
#         "2025-05-05-google-bert/bert-base-uncased-1e-05-data",
#         "2025-05-05-google-bert/bert-base-uncased-2e-05-data",
#         "2025-05-05-google-bert/bert-base-uncased-5e-05-data",
#         "2025-05-05-google-bert/bert-base-uncased-5e-06-data"
#     ]:
#     for epoch in [1,2,3,4,5]:
#         print(f"\n{folder_name}\n")
#         try:
#             d = DATA(folder_name, epoch=epoch, n_samples=1500)
            
#             t1 = time()
#             f1_macro = basicML(d)
#             t2 = time()

#             save.append({
#                 "filename" : folder_name,
#                 "n_samples" : 1500,
#                 "epoch" : epoch,
#                 "time" : t2-t1,
#                 "f1_macro" : float(f1_macro)
#             })
#             print((
#                 f"{'{}'.format(folder_name):<50}|"
#                 f"{'%.0f'%(1500):<10}|"
#                 f"{'%.0f'%(epoch):<10}|"
#                 f"{'%.2f'%(t2-t1):<10}|"
#                 f"{'%.3f'%(float(f1_macro)):<10}|"
#             ))
#         except Exception as e: 
#             save.append({
#                 "filename" : folder_name,
#                 "n_samples" : 1500,
#                 "epoch" : epoch,
#                 "time" : None,
#                 "f1_macro" : None
#             })
#             print((
#                 f"{'{}'.format(folder_name):<50}|"
#                 f"{'%.0f'%(1500):<10}|"
#                 f"{'%.0f'%(epoch):<10}|"
#                 f"{'FAILED':<10}|"
#                 f"{'FAILED':<10}|"
#                 f"\tError : {e}"
#             ))
# pd.DataFrame(save).to_csv("basicML.csv")
# CustomLogger().notify_when_done()