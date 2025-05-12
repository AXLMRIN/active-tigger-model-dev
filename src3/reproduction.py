# SKLEARN
import gc
import numpy as np
import pygad
from time import time
from torch import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import pandas as pd
from transformer_class import CustomLogger


from toolbox import routine, cMapper

routine(
    folder_name = "2025-05-05-answerdotai/ModernBERT-base-1e-05-data",
    classifier = RandomForestClassifier, 
    n_sample_range = [500, 750],
    epoch_range = [0,1,2,3,4,5],
    GA_parameters = {
        'num_genes' : 3,
        "gene_space" : [
            {'low' : 10, 'high' : 1000, 'step' : 50},
            [0,1,2],
            [30, 60, 90]
        ]
    },
    custom_mapping = cMapper(
        keys = ["n_estimators", "criterion","max_depth"],
        functions = [
            lambda x : int(x),
            lambda x : ["gini", "entropy", "log_loss"][x],
            lambda x : int(x)
        ] 
    )
)

# routineRandomForest("2025-05-05-answerdotai/ModernBERT-base-1e-05-data")
# routineRandomForest("2025-05-05-answerdotai/ModernBERT-base-2e-05-data")
# routineRandomForest("2025-05-05-answerdotai/ModernBERT-base-5e-05-data")
# routineRandomForest("2025-05-05-answerdotai/ModernBERT-base-5e-06-data")

# routineRandomForest("2025-05-05-FacebookAI/roberta-base-1e-05-data")
# routineRandomForest("2025-05-05-FacebookAI/roberta-base-2e-05-data")
# routineRandomForest("2025-05-05-FacebookAI/roberta-base-5e-05-data")
# routineRandomForest("2025-05-05-FacebookAI/roberta-base-5e-06-data")

# routineRandomForest("2025-05-05-google-bert/bert-base-uncased-1e-05-data")
# routineRandomForest("2025-05-05-google-bert/bert-base-uncased-2e-05-data")
# routineRandomForest("2025-05-05-google-bert/bert-base-uncased-5e-05-data")
# routineRandomForest("2025-05-05-google-bert/bert-base-uncased-5e-06-data")


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

# def routineKNN(folder_name : str) -> None:
    
#     def custom_mapping(idx, value, printFunction : bool = False):
#         if idx == 0:
#             if printFunction : return int(value)
#             else : return {"n_neighbors" : int(value)}
#         if idx == 1:
#             metric = ["cosine","l1","l2"]
#             if printFunction : return metric[value]
#             else : return {"metric" : metric[value]}
#         else:
#             raise(KeyError, "custom mapping idx not right")

#     save = []
#     fail = False
#     try : 
#         for n_samples in [250, 500, 750, 1000]: 
#             print(f"\nn_samples : {n_samples}\n")
#             for epoch in range(1,6):
#                 d, GA_param, t1, t2, optimum, value, optimizer = None,None,None,None,None,None,None 
#                 try : 
#                     d = DATA(folder_name,epoch, n_samples)
#                     GA_param = {
#                         'num_genes' : 2,
#                         "gene_space" : [
#                             {'low' : 1, 'high' : 20},
#                             [0,1,2]
#                         ],
#                         "gene_type": [int,int],
#                     }
#                     classifier = KNeighborsClassifier
#                     param_mapping = custom_mapping

#                     optimizer = optimize_classifier(d, classifier, GA_param, param_mapping)

#                     t1 = time()
#                     optimum, value  = optimizer.run()
#                     t2 = time()

#                     save.append({
#                         "filename" : folder_name,
#                         "n_samples" : n_samples,
#                         "epoch" : epoch,
#                         "time" : t2-t1,
#                         "f1_macro" : float(value),
#                         "n_neighbors" : custom_mapping(0, optimum[0],True),
#                         "metric" : custom_mapping(1, optimum[1],True)
#                     })
#                     print((
#                         f"{'%.0f'%(n_samples):<10}|"
#                         f"{'%.0f'%(epoch):<10}|"
#                         f"{'%.2f'%(t2-t1):<10}|"
#                         f"{'%.3f'%(float(value)):<10}|"
#                         f"{'{}'.format(custom_mapping(0, optimum[0],True)):<10}|"
#                         f"{'{}'.format(custom_mapping(1, optimum[1],True)):<10}|"
#                     ))

#                 except Exception as e: 
#                     save.append({
#                         "filename" : folder_name,
#                         "n_samples" : n_samples,
#                         "epoch" : epoch,
#                         "time" : None,
#                         "f1_macro" : None,
#                         "n_neighbors" : None,
#                         "metric" : None
#                     })
#                     print((
#                         f"{'%.0f'%(n_samples):<10}|"
#                         f"{'%.0f'%(epoch):<10}|"
#                         f"{'FAILED':<10}|"
#                         f"{'FAILED':<10}|"
#                         f"{'FAILED':<10}|"
#                         f"{'FAILED':<10}|"
#                         f"\tError : {e}"
#                     ))

#                 finally : 
#                     del d, GA_param, t1, t2, optimum, value, optimizer
#                     gc.collect()
        
#     except : 
#         fail = True

#     finally : 
#         df = pd.read_csv("KNN.csv")
#         df = pd.concat((df,pd.DataFrame(save)))
#         df.to_csv("KNN.csv", index = False)
#         CustomLogger().notify_when_done(f"KNN {folder_name}")

# routineKNN("2025-05-05-answerdotai/ModernBERT-base-1e-05-data")
# routineKNN("2025-05-05-answerdotai/ModernBERT-base-2e-05-data")
# routineKNN("2025-05-05-answerdotai/ModernBERT-base-5e-05-data")
# routineKNN("2025-05-05-answerdotai/ModernBERT-base-5e-06-data")

# routineKNN("2025-05-05-FacebookAI/roberta-base-1e-05-data")
# routineKNN("2025-05-05-FacebookAI/roberta-base-2e-05-data")
# routineKNN("2025-05-05-FacebookAI/roberta-base-5e-05-data")
# routineKNN("2025-05-05-FacebookAI/roberta-base-5e-06-data")

# routineKNN("2025-05-05-google-bert/bert-base-uncased-1e-05-data")
# routineKNN("2025-05-05-google-bert/bert-base-uncased-2e-05-data")
# routineKNN("2025-05-05-google-bert/bert-base-uncased-5e-05-data")
# routineKNN("2025-05-05-google-bert/bert-base-uncased-5e-06-data")