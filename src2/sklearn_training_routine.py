from toolbox.CustomLogger import CustomLogger
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import plotly.express as px
import pandas as pd
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
    print(epochs)
    break
    for epoch_file in os.listdir(f"./sklearn_save/{model}"):
        print(epoch_file)
        X_train = load(f"./sklearn_save/{model}/{epoch_file}", weights_only=True).numpy()[:,:-2]
        y_train = load(f"./sklearn_save/{model}/{epoch_file}",weights_only=True).numpy()[:,-2]
        
        X_test = load(f"./sklearn_save/{model}/{epoch_file}", weights_only=True).numpy()[:,:-2]
        y_test = load(f"./sklearn_save/{model}/{epoch_file}",weights_only=True).numpy()[:,-2]

        print(X.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Adaboost-----------------------------------------------------------------------------------
        print("Adaboost")
        for n in [400,450,500,800] : 
            clf = AdaBoostClassifier(n_estimators=n, random_state=0)
            clf.fit(X_train, y_train)
        
            ababoost.append({
                "n_estimators" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch_file)
            })
        # RandomForestClassifier-----------------------------------------------------------------------------------
        print("RandomForestClassifier")
        for n in [20,40,60] : 
            clf = RandomForestClassifier(max_depth=n, random_state=0)
            clf.fit(X_train, y_train)
        
            random_forest.append({
                "max_depth" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch_file)
            })
        
        # NeuralNet-----------------------------------------------------------------------------------
        print("NeuralNet")
        for n in [(5,),(10,),(15,),(200,5)] : 
            clf = MLPClassifier(hidden_layer_sizes=n, max_iter = 300)
            clf.fit(X_train, y_train)
            
            neural_net.append({
                "hidden_layers" : str(n),
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch_file)
            })
        # GaussianProcessClassifier-----------------------------------------------------------------------------------
        print("GaussianProcessClassifier")
        for n in [0] : 
            clf = GaussianProcessClassifier()
            clf.fit(X_train, y_train)
        
            gaussian.append({
                "unnamed" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch_file)
            })
        
        
        # KNN-----------------------------------------------------------------------------------
        print("KNN")
        for n in [3,5,10,20] : 
            clf = KNeighborsClassifier(n_neighbors = n)
            clf.fit(X_train, y_train)
        
            knn.append({
                "n_neighbors" : n,
                "score" : clf.score(X_test,y_test),
                "epoch" : str(epoch_file)
            })
    
    
    px.scatter(
            pd.DataFrame(ababoost), 
            x = 'n_estimators', 
            y = "score", 
            color = "epoch", 
            mode = "lines+markers"
        ).write_html(f"./sklearn_save/{model}/adaboost.html")
    px.scatter(
            pd.DataFrame(random_forest), 
            x = 'max_depth', 
            y = "score", 
            color = "epoch", 
            mode = "lines+markers"
        ).write_html(f"./sklearn_save/{model}/random_forest.html")
    px.scatter(
            pd.DataFrame(neural_net), 
            x = 'hidden_layers', 
            y = "score", 
            color = "epoch", 
            mode = "lines+markers"
        ).write_html(f"./sklearn_save/{model}/neural_net.html")
    px.scatter(
            pd.DataFrame(gaussian), 
            x = 'unnamed', 
            y = "score", 
            color = "epoch", 
            mode = "lines+markers"
        ).write_html(f"./sklearn_save/{model}/gaussian.html")
    px.scatter(
            pd.DataFrame(knn), 
            x = 'n_neighbors', 
            y = "score", 
            color = "epoch", 
            mode = "lines+markers"
        ).write_html(f"./sklearn_save/{model}/knn.html")
    
    
# CustomLogger().notify_when_done()