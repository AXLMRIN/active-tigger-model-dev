from . import DATA
from sklearn.metrics import f1_score
import pygad
from time import time

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
            "parallel_processing" : 5, 
            
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
    
    def run(self) -> tuple[float,float,float]:
        t1 = time()
        instance = pygad.GA(**self.GA_parameters)
        instance.run()
        t2 = time()
        optimum, value, _ = instance.best_solution()
        return optimum, value, t2-t1, instance.generations_completed