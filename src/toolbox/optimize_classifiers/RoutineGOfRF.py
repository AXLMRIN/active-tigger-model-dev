# IMPORTS ######################################################################
from typing import Any
from ..CustomLogger import CustomLogger
from .RoutineGOfSC import RoutineGOfSC
from sklearn.ensemble import RandomForestClassifier
# SCRIPTS ######################################################################
class RoutineGOfRF(RoutineGOfSC) : 
    """
    """
    def __init__(self, 
        foldername : str, 
        ranges_of_configs : dict[str:list[Any]],
        n_samples : int,
        logger : CustomLogger,
        extra_GA_parameters : dict = {}
        ) -> None:
        """
        """
        super().__init__(
            foldername = foldername,
            ranges_of_configs = ranges_of_configs,
            n_samples = n_samples,
            extra_GA_parameters = extra_GA_parameters,

            classifier = RandomForestClassifier,
            parameters_mapper = {
                "n_estimators" : self.__n_estimators_mapper_function,
                # "criterion" : self.__criterion_mapper_function,
                # "max_depth" : self.__max_depth_mapper_function
            },
            gene_space = {
                'num_genes' : 1,
                "gene_space" : [
                    {'low' : 10, 'high' : 1000, 'step' : 110},
                    # [0,1,2],
                    # [30, 60, 90]
                ]
            },

            logger = logger
        )

    def __n_estimators_mapper_function(self, value):
        return int(value)
    
    def __criterion_mapper_function(self, value):
        crits = ["gini", "entropy", "log_loss"]
        return crits[int(value)]
    
    def __max_depth_mapper_function(self, value):
        return int(value)
    
    def routine(self,filename : str, n_iterations : int) -> None :
        super().routine(filename,n_iterations)