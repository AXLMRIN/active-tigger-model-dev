# SKLEARN
import gc
import numpy as np
import pygad
from time import time
from torch import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample
import pandas as pd
from transformer_class import CustomLogger

class DATA:
    def __init__(self, foldername_root : str, epoch : int, n_samples : int = 500):
        self.X_train = load(f"./{foldername_root}/epoch_{epoch}/train_embedded.pt",
            weights_only=True).cpu().numpy()
        labels = load(f"./{foldername_root}/epoch_{epoch}/train_labels.pt",
            weights_only=True).cpu().numpy()
        self.y_train = [np.argmax(row).item() for row in labels]
        self.X_train, self.y_train = resample(self.X_train,self.y_train, n_samples=n_samples)

        self.X_test = load(f"./{foldername_root}/epoch_{epoch}/test_embedded.pt",
            weights_only=True).cpu().numpy()
        labels = load(f"./{foldername_root}/epoch_{epoch}/test_labels.pt",
            weights_only=True).cpu().numpy()
        self.y_test = [np.argmax(row).item() for row in labels]
        self.X_test, self.y_test = resample(self.X_test,self.y_test)

class optimize_classifier:
    def __init__(self, d : DATA, classifier, GA_param : dict, param_mapping):
        self.d = d
        self.classifier = classifier
        self.param_mapping = param_mapping
        self.GA_parameters = {
            #Must Specify
            'fitness_func' : self.fitness_function,
            'num_generations' : 50,
            
            'sol_per_pop' : 12,
            'num_parents_mating' : 3,
            'keep_elitism' : 2,
            
            **GA_param,
            
            "stop_criteria" : "saturate_5",
            
            # Default
            'mutation_type' : "random",
            'parent_selection_type' : "sss",
            'crossover_type' : "single_point",
            'mutation_percent_genes' : 50,
            # Other
            'save_solutions' : False,
        }
    
    def fitness_function(self, GAI, SOL, SOLIDX):
        params = {}
        for idx,value in enumerate(SOL):
            params = {**params, **self.param_mapping(idx, value)}
        clf = self.classifier(**params)
        clf.fit(self.d.X_train, self.d.y_train)
        
        return f1_score(y_true=self.d.y_test, y_pred=clf.predict(self.d.X_test), average='macro')
    
    def run(self):
        instance = pygad.GA(**self.GA_parameters)
        instance.run()
        return instance.best_solution()[0:2]


def routineRandomForest(folder_name : str) -> None:
    
    def custom_mapping(idx, value, printFunction : bool = False):
        if idx == 0:
            if printFunction : return int(value)
            else : return {"n_estimators" : int(value)}
        elif idx == 1:
            criterion = ["gini", "entropy", "log_loss"]
            if printFunction : return criterion[value]
            else :return {"criterion" : criterion[value]}
        elif idx == 2:
            if printFunction : return (int(value))
            else : return {"max_depth" : int(value)}
        else:
            raise(KeyError, "custom mapping idx not right")

    save = []
    fail = False
    try : 
        for n_samples in [500, 750]: 
            print(f"\nn_samples : {n_samples}\n")
            for epoch in range(2,5):
                d, GA_param, t1, t2, optimum, value, optimizer = None,None,None,None,None,None,None 
                try : 
                    d = DATA(folder_name,epoch, n_samples)
                    GA_param = {
                        'num_genes' : 3,
                        "gene_space" : [
                            {'low' : 10, 'high' : 1000, 'step' : 50},
                            [0,1,2],
                            [30, 60, 90]
                        ],
                        "gene_type": [int,int, int],
                    }
                    classifier = RandomForestClassifier
                    param_mapping = custom_mapping

                    optimizer = optimize_classifier(d, classifier, GA_param, param_mapping)

                    t1 = time()
                    optimum, value  = optimizer.run()
                    t2 = time()

                    save.append({
                        "filename" : folder_name,
                        "n_samples" : n_samples,
                        "epoch" : epoch,
                        "time" : t2-t1,
                        "f1_macro" : float(value),
                        "n_estimators" : custom_mapping(0, optimum[0],True),
                        "criterion" : custom_mapping(1, optimum[1],True),
                        "max_depth" : custom_mapping(2, optimum[2],True)
                    })
                    print((
                        f"{'%.0f'%(n_samples):<10}|"
                        f"{'%.0f'%(epoch):<10}|"
                        f"{'%.2f'%(t2-t1):<10}|"
                        f"{'%.3f'%(float(value)):<10}|"
                        f"{'{}'.format(custom_mapping(0, optimum[0],True)):<10}|"
                        f"{'{}'.format(custom_mapping(1, optimum[1],True)):<10}|"
                        f"{'{}'.format(custom_mapping(2, optimum[2],True)):<10}|"
                    ))

                except Exception as e: 
                    save.append({
                        "filename" : "2025-05-05-answerdotai/ModernBERT-base-1e-05-data",
                        "n_samples" : n_samples,
                        "epoch" : epoch,
                        "time" : None,
                        "f1_macro" : None,
                        "n_estimators" : None,
                        "criterion" : None,
                        "max_depth" : None
                    })
                    print((
                        f"{'%.0f'%(n_samples):<10}|"
                        f"{'%.0f'%(epoch):<10}|"
                        f"{'FAILED':<10}|"
                        f"{'FAILED':<10}|"
                        f"{'FAILED':<10}|"
                        f"{'FAILED':<10}|"
                        f"{'FAILED':<10}|"
                        f"\tError : {e}"
                    ))

                finally : 
                    del d, GA_param, t1, t2, optimum, value, optimizer
                    gc.collect()
        
    except : 
        fail = True

    finally : 
        df = pd.read_csv("RandomForest.csv")
        df = pd.concat((df,pd.DataFrame(save)))
        df.to_csv("RandomForest.csv", index = False)
        CustomLogger().notify_when_done(f"fail : {fail}")

routineRandomForest("2025-05-05-answerdotai/ModernBERT-base-5e-06-data")

routineRandomForest("2025-05-05-FacebookAI/roberta-base-1e-05-data")
routineRandomForest("2025-05-05-FacebookAI/roberta-base-2e-05-data")
routineRandomForest("2025-05-05-FacebookAI/roberta-base-5e-05-data")
routineRandomForest("2025-05-05-FacebookAI/roberta-base-5e-06-data")

routineRandomForest("2025-05-05-google-bert/bert-base-uncased-1e-05-data")
routineRandomForest("2025-05-05-google-bert/bert-base-uncased-2e-05-data")
routineRandomForest("2025-05-05-google-bert/bert-base-uncased-5e-05-data")
routineRandomForest("2025-05-05-google-bert/bert-base-uncased-5e-06-data")