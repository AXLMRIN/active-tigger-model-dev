# IMPORTS ######################################################################
from typing import Callable, Any
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
    """GeneticOptimiserForSklearnClassifier is a routine using the PyGAD library
    to optimise the hyperparameters for Scikit-Learn classifiers such as 
    KKNeighborsClassifierNN or RandomForestClassifier. The "loss function" that is
    being optimised is the f1 macro tested on the test set for classifier 
    fitted on the train set.

    the main features and functions are:
        - collects the data, classifier and pygad parameters
        - create a pygad instance and run it
        - return the f1 score macro

    Possible UPGRADE: change the metric.
    """
    def __init__(self, 
        data : DataHandlerForGOfSC,
        classifier, 
        parameters_mapper : dict[str:Callable[[float],Any]],
        gene_space : dict[str:list[Any]|dict[str:Any]],
        extra_GA_parameters : dict [str:Any]= {}
        ) -> None: 
        """Builds the GeneticOptimiserForSklearnClassifier object.

        Parameters:
        -----------
            - data (DataHandlerForGOfSC): train and test set for the classifier.
            - classifier: scikit-learn classifier (any object that has a fit and 
                predict method)
            - parameters_mapper (dict[str:Callable[[float],Any]]): a dictionnary
                binding the names of the classifier's parameters and functions 
                that adapt the PyGAD optimisation space to the classifier __init__.
                    Explenation: the PyGAD optimisation space is a np.ndarray of 
                        floats but the classifier requires ints or strings. 
            - gene_space (dict): dictionnary defining the gene space such as defined
                in the PyGAD documentation. Also includes the number of genes 
                (num_genes) from the PyGAD documentation
            - extra_GA_parameters (dict, default = {}): extra GA parameters 
                (overwrite the default parameters) see PyGAD documentation.
        
        Returns:
        --------
            /

        Initialised variables:
        ----------------------
            - self.__data (DataHandlerForGOfSC): train and test set for the 
                classifier.
            - self.__classifier: scikit-learn classifier (any object that has a fit and 
                predict method)
            - self.__hyper_parameters_mapper_keys (list[str]): the list of the names 
                of hyperparameters (from parameters_mapper)
            - self.__hyper_parameters_mapper_functions (list[Callable[[float],Any]]):
                the list of mapping functions (from parameters_mapper)
            - self.GA_instance_parameters (dict[str:Any]): the parameters for 
                the GA object.
        """
        self.__data : DataHandlerForGOfSC = data
        self.__classifier = classifier
        self.__hyper_parameters_mapper_keys : list[str]= \
            list(parameters_mapper.keys())
        self.__hyper_parameters_mapper_functions : list[Callable[[float],Any]] = \
            list(parameters_mapper.values())

        # GA parameters
        # Deduced from my own experience, can be overwritten by extra_GA_parameters
        num_genes : int = gene_space["num_genes"]
        deduced_parameters = {
            'fitness_func' : self.fitness_func,
            'sol_per_pop' : int(4 * num_genes),
            'keep_elitism' : int(max(0.5 * 0.5 * 4 * num_genes, 2)),
            'num_parents_mating' : int(max(0.2 * 4 * num_genes, 1))
        }

        self.GA_instance_parameters = merge(
            DEFAULT_GA_PARAMETERS, 
            deduced_parameters,
            gene_space,
            extra_GA_parameters
        )
    
    def __parameter_value_binder(self,idx, value : Any) -> dict[int:Any] :
        """Simple function returning a dictionnary with the name of the parameter 
        and its value after calling the corresponding mapping function.

        Parameters:
        -----------
            - idx (int): the index of the parameter in 
                self.__hyper_parameters_mapper_keys and 
                self.__hyper_parameters_mapper_functions
            - value (Any): the value to pass to the mapping function.

        Returns:
        --------
            - dict: dictionnary binding the parameter name and the value returned
                by the mapping function.
        """
        parameter_name : str = self.__hyper_parameters_mapper_keys[idx]
        function_to_apply : Any = self.__hyper_parameters_mapper_functions[idx]
        return {parameter_name : function_to_apply(value)}

    def __make_parameters_out_of_SOL(self, SOL : np.ndarray) -> dict[str:Any] :
        """From the SOL object in the PyGAD pipeline, which correspond to a solution.
        Thus it is an array of n_genes objects. This functions takes this array 
        and returns a dictionnary of all the parameters name and the value returned
        by the corresponding mapping function.

        Parameters:
        -----------
            - SOL (np.ndarray): the array of n_genes object corresponding to a 
                solution.
        
        Returns:
        --------
            - dict: dictionnary of the n_genes parameters name bond to the value 
                returned by the mapping function.
        """ 
        params = {}
        for idx, value in enumerate(SOL):
            params = merge(params, self.__parameter_value_binder(idx,value))
        return params
    
    def fitness_func(self, GAI, SOL : np.ndarray, SOLIDX) -> float :
        """This is the function called by the PyGAD instance. It creates a 
        classifier with parameters defined by the SOL object, then fit the classifier
        on the data and finally return the F1 score.

        Parameters:
        -----------
            - GAI: not used, required to create the PyGAD instance.
            - SOL (np.ndarray): the array of n_genes object corresponding to a 
                solution.
            - SOLIDX: not used, required to create the PyGAD instance.

        Returns:
        --------
            - float: the f1 score (macro) of the classifier fitted on the data.
        """
        params = self.__make_parameters_out_of_SOL(SOL)
        clf = self.__classifier(**params)
        clf.fit(self.__data.X_train, self.__data.y_train)

        y_pred : np.ndarray = clf.predict(self.__data.X_test)
        y_true : np.ndarray = self.__data.y_test
        return f1_score(y_true, y_pred, average='macro')
    
    def run(self) -> tuple[dict[str:Any],float,float,int]:
        """Creates a PyGAD instance to optimise the hyperparameters and runs the 
        optimisation. Returns the optimised hyperparameters, the maximum f1 macro,
        the optimisation time and the number of generations required to reach this
        optimum. 

        Parameters:
        -----------
            /
        
        Returns:
        --------
            - dict: dictionnary of the parameters' name and their optimised values.
            - float: the maximum f1 macro.
            - float: the optimisation time.
            - float: the number of generation ran to reach this optimum.
        """
        t1 = time()
        instance = pygad.GA(**self.GA_instance_parameters)
        instance.run()
        t2 = time()
        optimum, value, _ = instance.best_solution()
        number_of_completed_generations : int = instance.generations_completed
        
        # Format outputs
        zipped = zip(self.__hyper_parameters_mapper_keys, self.__hyper_parameters_mapper_functions,
                      optimum)
        optimum : dict[str:Any] = {
            key : mapper(value) for key, mapper, value in zipped
        }
        value = float(value) 
        
        return optimum, value, t2-t1, number_of_completed_generations
