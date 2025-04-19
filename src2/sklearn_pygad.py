import pygad
from torch import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample, shuffle

# lexic
# 0 : epoch
# 1 : Max depth
# 2 : Number of elements

class DATA:
    def __init__(self) -> None:
        self.X_train = {
            epoch : load(("sklearn_save_test/answerdotai_ModernBERT-base/"
                          f"epoch_{epoch}_train.pt"), weights_only = True).\
                          numpy()[:,1:-1]
            for epoch in range(-1,5)
        }

        self.y_train = {
            epoch : load(("sklearn_save_test/answerdotai_ModernBERT-base/"
                          f"epoch_{epoch}_train.pt"), weights_only = True).\
                          numpy()[:,-1]
            for epoch in range(-1,5)
        }
        
        self.X_test = {
            epoch : load(("sklearn_save_test/answerdotai_ModernBERT-base/"
                          f"epoch_{epoch}_test.pt"), weights_only = True).\
                          numpy()[:,1:-1]
            for epoch in range(-1,5)
        }
        
        self.y_test = {
            epoch : load(("sklearn_save_test/answerdotai_ModernBERT-base/"
                          f"epoch_{epoch}_test.pt"), weights_only = True).\
                          numpy()[:,-1]
            for epoch in range(-1,5)
        }
    
    def get_epoch(self, epoch : int) \
        -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] :
        return (
            self.X_train[epoch],
            self.y_train[epoch],
            self.X_test[epoch],
            self.y_test[epoch]
        )

data = DATA()
data.get_epoch(-1)

def fitness_func(ga_instance, solution : np.ndarray, solution_idx : int):
    X_train, y_train, X_test, y_test = data.get_epoch(solution[0])
    # Upsampling
    X_train_split = {
        int(label) : X_train[y_train == label,:]
        for label in set(y_train)
    }

    for label in X_train_split.keys():
        X_train_split[label] = resample(X_train_split[label],
                                        n_samples=solution[2])
    
    X_train_resampled = np.concatenate(tuple(X_train_split.values()))
    y_train_resampled = np.concatenate(tuple(
        label * np.ones(solution[2]) for label in X_train_split.keys()
    ))
    X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled)
    clf = RandomForestClassifier(max_depth=solution[1], random_state=0)
    clf.fit(X_train_resampled, y_train_resampled)
    score_model = clf.score(X_test, y_test) # [0,1]
    score_sample = solution[2] / 1000 # [1, inf[
    return score_model - 0.025 * score_sample

GA_parameters = {
    #Must Specify
    'fitness_func' : fitness_func,
    'num_generations' : 5,
    
    'num_parents_mating' : 5,
    'sol_per_pop' : 15,
    'num_genes' : 3,
    
    'keep_elitism' : 2,
    "gene_space" : [
        {'low' : -1, 'high' : 4},
        {'low' : 1, 'high' : 60},
        {'low' : 50, 'high' : 1500, 'step' : 50}
    ],
    "gene_type": [int,int, int],
    "stop_criteria" : "saturate_5",
    # Default
    'mutation_type' : "random",
    'parent_selection_type' : "sss",
    'crossover_type' : "single_point",
    'mutation_percent_genes' : 50,
    # Other
    'save_solutions' : False,
}
ga_instance = pygad.GA(**GA_parameters)
ga_instance.run()
print(ga_instance.best_solution())