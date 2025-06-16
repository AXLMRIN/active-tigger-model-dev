# IMPORTS ######################################################################
from typing import Any
from .RoutineGOfSC import RoutineGOfSC
from sklearn.neighbors import KNeighborsClassifier
# SCRIPTS ######################################################################
class RoutineGOfKNN(RoutineGOfSC) : 
    """
    """
    def __init__(self, 
        foldername : str, 
        ranges_of_configs : dict[str:list[Any]],
        n_samples : int,
        extra_GA_parameters : dict = {}
        ) -> None:
        """
        """
        super().__init__(
            foldername = foldername,
            ranges_of_configs = ranges_of_configs,
            n_samples = n_samples,
            extra_GA_parameters = extra_GA_parameters,

            classifier = KNeighborsClassifier,
            parameters_mapper = {
                "n_neighbors" : self.__n_neighbors_mapper_function,
                "metric" : self.__metric_mapper_function
            },
            gene_space = {
                'num_genes' : 2,
                "gene_space" : [
                    {'low' : 1, 'high' : 20},
                    [0,1,2]
                ]
            },

        )
    def __n_neighbors_mapper_function(self, value):
        return int(value)

    def __metric_mapper_function(self, idx):
        crits = ["cosine","l1","l2"]
        return crits[int(idx)]
    
    def routine(self,filename : str, n_iterations : int) -> None :
        super().routine(filename,n_iterations)