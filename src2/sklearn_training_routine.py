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
        

        shuffle_index_train = np.random.shuffle(np.arange(np.shape(X_train)[0]))
        X_train = X_train[shuffle_index_train,:]
        y_train = y_train[shuffle_index_train]
        
        shuffle_index_test = np.random.shuffle(np.arange(np.shape(X_test)[0]))
        X_test = X_test[shuffle_index_test,:]
        y_test = y_test[shuffle_index_test]

        # debug
        X_train = X_train[:100,:100]
        y_train = y_train[:100]

        X_test = X_test[:50,:100]
        y_test = y_test[:50]

        # Adaboost-----------------------------------------------------------------------------------
        print("Adaboost")
        for n in [400,450,500,800] : 
            clf = AdaBoostClassifier(n_estimators=n, random_state=0)
            clf.fit(X_train, y_train)
        
            ababoost.append({
                "n_estimators" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch)
            })
        # RandomForestClassifier-----------------------------------------------------------------------------------
        print("RandomForestClassifier")
        for n in [20,40,60] : 
            clf = RandomForestClassifier(max_depth=n, random_state=0)
            clf.fit(X_train, y_train)
        
            random_forest.append({
                "max_depth" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch)
            })
        
        # NeuralNet-----------------------------------------------------------------------------------
        print("NeuralNet")
        for n in [(5,),(10,),(15,),(200,5)] : 
            clf = MLPClassifier(hidden_layer_sizes=n, max_iter = 300)
            clf.fit(X_train, y_train)
            
            neural_net.append({
                "hidden_layers" : str(n),
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch)
            })
        # GaussianProcessClassifier-----------------------------------------------------------------------------------
        print("GaussianProcessClassifier")
        for n in [0] : 
            clf = GaussianProcessClassifier()
            clf.fit(X_train, y_train)
        
            gaussian.append({
                "unnamed" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch)
            })
        
        
        # KNN-----------------------------------------------------------------------------------
        print("KNN")
        for n in [3,5,10,20] : 
            clf = KNeighborsClassifier(n_neighbors = n)
            clf.fit(X_train, y_train)
        
            knn.append({
                "n_neighbors" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch)
            })
    
    
    px.line(
            pd.DataFrame(ababoost), 
            x = 'n_estimators', 
            y = "score", 
            color = "epoch"
        ).write_html(f"./sklearn_save/{model}/adaboost.html")
    px.line(
            pd.DataFrame(random_forest), 
            x = 'max_depth', 
            y = "score", 
            color = "epoch"
        ).write_html(f"./sklearn_save/{model}/random_forest.html")
    px.line(
            pd.DataFrame(neural_net), 
            x = 'hidden_layers', 
            y = "score", 
            color = "epoch"
        ).write_html(f"./sklearn_save/{model}/neural_net.html")
    px.line(
            pd.DataFrame(gaussian), 
            x = 'unnamed', 
            y = "score", 
            color = "epoch"
        ).write_html(f"./sklearn_save/{model}/gaussian.html")
    px.line(
            pd.DataFrame(knn), 
            x = 'n_neighbors', 
            y = "score", 
            color = "epoch"
        ).write_html(f"./sklearn_save/{model}/knn.html")
    break

CustomLogger().notify_when_done()