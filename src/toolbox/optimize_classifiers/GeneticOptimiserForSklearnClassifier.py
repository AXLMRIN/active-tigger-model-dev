# IMPORTS ######################################################################
from sklearn.metrics import f1_score
import pygad
from time import time
from mergedeep import merge
import numpy as np
from typing import Any, Callable
from .DataHandlerForGOfSC import DataHandlerForGOfSC
# CONSTANTS ####################################################################
DEFAULT_GA_PARAMETERS : dict = {
    #Must Specify
    'num_generations' : 50,
    
    "stop_criteria" : "saturate_5",
    
    # Default
    'mutation_type' : "random",
    'parent_selection_type' : "sss",
    'crossover_type' : "single_point",
    'mutation_percent_genes' : 50,
    # Other
    'save_solutions' : False,
}
# SCRIPTS ######################################################################
class GeneticOptimiserForSklearnClassifier :
    """
    """
    def __init__(self, 
        data : DataHandlerForGOfSC,
        classifier, 
        parameters_mapper : dict,
        gene_space : dict,
        extra_GA_parameters : dict = {}) -> None: 
        """
        """
        self.__data = data
        self.__classifier = classifier
        self.__parameters_mapper_keys : list[str]= \
            list(parameters_mapper.keys())
        self.__parameters_mapper_functions : list[Callable] = \
            list(parameters_mapper.values())
        self.__num_genes = gene_space["num_genes"]

        # GA parameters
        deduced_parameters = {
            'fitness_func' : self.fitness_func,
            'sol_per_pop' : int(4 * self.__num_genes),
            'keep_elitism' : int(max(0.5 * 0.5 * 4 * self.__num_genes, 2)),
            'num_parents_mating' : int(max(0.2 * 4 * self.__num_genes, 1))
        }

        self.GA_instance_parameters = merge(
            DEFAULT_GA_PARAMETERS, 
            deduced_parameters,
            gene_space,
            extra_GA_parameters
        )
    
    def __parameter_value_binder(self,idx, value : Any) -> dict[int:Any] :
        """
        """
        parameter_name : str = self.__parameters_mapper_keys[idx]
        function_to_apply : Any = self.__parameters_mapper_functions[idx]
        return {parameter_name : function_to_apply(value)}

    def __make_parameters_out_of_SOL(self, SOL : np.ndarray) -> dict[str:Any] :
        """
        """ 
        params = {}
        for idx, value in enumerate(SOL):
            params = merge(params, self.__parameter_value_binder(idx,value))
        return params
    
    def fitness_func(self, GAI, SOL : np.ndarray, SOLIDX) -> float :
        """
        """
        params = self.__make_parameters_out_of_SOL(SOL)
        clf = self.__classifier(**params)
        clf.fit(self.__data.X_train, self.__data.y_train)

        y_pred : np.ndarray = clf.predict(self.__data.X_test)
        y_true : np.ndarray = self.__data.y_test
        return f1_score(y_true, y_pred, average='macro')
    
    def run(self) -> tuple[dict[str:Any],float,float,int]:
        """
        """
        t1 = time()
        instance = pygad.GA(**self.GA_instance_parameters)
        instance.run()
        t2 = time()
        optimum, value, _ = instance.best_solution()
        number_of_completed_generations : int = instance.generations_completed
        
        # Format outputs
        zipped = zip(self.__parameters_mapper_keys, self.__parameters_mapper_functions,
                      optimum)
        optimum : dict[str:Any] = {
            key : mapper(value) for key, mapper, value in zipped
        }
        value = float(value) 
        
        return optimum, value, t2-t1, number_of_completed_generations
