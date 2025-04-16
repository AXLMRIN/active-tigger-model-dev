from toolbox.CustomLogger import CustomLogger
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch import load


for model in os.listdir("./sklearn_save"):
    print(model)
    ababoost = []
    random_forest = []
    neural_net = []
    gaussian = []
    knn = []
    epochs = []
    for epoch_file in os.listdir(f"./sklearn_save/{model}"):
        # epoch files are written like epoch_xx_{train|test}.pt
        if epoch_file.startswith("epoch_")&epoch_file.endswith(".pt"): 
            epoch_id = epoch_file.split("_")[1]
            if epoch_id not in epochs : 
                epochs.append(epoch_id)
    epochs = sorted(epochs)
    
    del epoch_file, epoch_id

    for epoch in epochs:
        print(epoch)
        X_train = load(f"./sklearn_save/{model}/epoch_{epoch}_train.pt", weights_only=True).\
                    numpy()[:,1:-1]
        y_train = load(f"./sklearn_save/{model}/epoch_{epoch}_train.pt",weights_only=True).\
                    numpy()[:,-1]
        print(X_train.shape)
        print(y_train.shape)
        
        X_test = load(f"./sklearn_save/{model}/epoch_{epoch}_test.pt", weights_only=True)\
                    .numpy()[:,1:-1]
        y_test = load(f"./sklearn_save/{model}/epoch_{epoch}_test.pt",weights_only=True)\
                    .numpy()[:,-1]
        print(X_test.shape)
        print(y_test.shape)
        
        # Rows are already shuffled

        # debug
        X_train = X_train[:100,:100]
        y_train = y_train[:100]

        X_test = X_test[:50,:100]
        y_test = y_test[:50]

        # Adaboost-----------------------------------------------------------------------------------
        print("Adaboost")
        clf = AdaBoostClassifier(n_estimators=400, random_state=0)
        clf.fit(X_train, y_train)
    
        ababoost.append({
            "n_estimators" : '400',
            "score" : clf.score(X_test,y_test),
            "epoch" : int(epoch)
        })
        # RandomForestClassifier-----------------------------------------------------------------------------------
        print("RandomForestClassifier")
        clf = RandomForestClassifier(max_depth=60, random_state=0)
        clf.fit(X_train, y_train)
    
        random_forest.append({
            "max_depth" : '60',
            "score" : clf.score(X_test,y_test),
            "epoch" : int(epoch)
        })
        
        # NeuralNet-----------------------------------------------------------------------------------
        print("NeuralNet")
        clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter = 300)
        clf.fit(X_train, y_train)
        
        neural_net.append({
            "hidden_layers" : str((5,)),
            "score" : clf.score(X_test,y_test),
            "epoch" : int(epoch)
        })

        # GaussianProcessClassifier-----------------------------------------------------------------------------------
        print("GaussianProcessClassifier") 
        clf = GaussianProcessClassifier()
        clf.fit(X_train, y_train)
    
        gaussian.append({
            "unnamed" : "default",
            "score" : clf.score(X_test,y_test),
            "epoch" : int(epoch)
        })
    
        
        # KNN-----------------------------------------------------------------------------------
        print("KNN")
        clf = KNeighborsClassifier(n_neighbors = 10)
        clf.fit(X_train, y_train)
    
        knn.append({
            "n_neighbors" : '10',
            "score" : clf.score(X_test,y_test),
            "epoch" : int(epoch)
        })
    
    
    px.line(
            pd.DataFrame(ababoost), 
            x = 'epoch', 
            y = "score", 
            color = "n_estimators"
        ).write_html(f"./sklearn_save/{model}/adaboost.html")
    px.line(
            pd.DataFrame(random_forest), 
            x = 'epoch', 
            y = "score", 
            color = "max_depth"
        ).write_html(f"./sklearn_save/{model}/random_forest.html")
    px.line(
            pd.DataFrame(neural_net), 
            x = 'epoch', 
            y = "score", 
            color = "hidden_layers"
        ).write_html(f"./sklearn_save/{model}/neural_net.html")
    px.line(
            pd.DataFrame(gaussian), 
            x = 'epoch', 
            y = "score", 
            color = "unnamed"
        ).write_html(f"./sklearn_save/{model}/gaussian.html")
    px.line(
            pd.DataFrame(knn), 
            x = 'epoch', 
            y = "score", 
            color = "n_neighbors"
        ).write_html(f"./sklearn_save/{model}/knn.html")
    break

CustomLogger().notify_when_done()