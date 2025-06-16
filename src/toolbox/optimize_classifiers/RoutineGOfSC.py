# IMPORTS ######################################################################
from .GeneticOptimiserForSklearnClassifier import GeneticOptimiserForSklearnClassifier
from .DataHandlerForGOfSC import DataHandlerForGOfSC
from typing import Any
from itertools import product
import os
from .. import ROOT_MODELS, ROOT_RESULTS
from ..general import clean
from torch import load
import pandas as pd
import numpy as np
# SCRIPTS ######################################################################
class RoutineGOfSC:
    """
    """
    def __init__(self, 
        foldername : str, 
        ranges_of_parameters : dict[str:list[Any]], 
        classifier, 
        parameters_mapper : dict,
        gene_space : dict,
        extra_GA_parameters : dict = {}) -> None:
        """
        """
        self.__foldername : str = foldername
        self.__ranges_of_parameters : dict[str:list] = ranges_of_parameters
        self.__classifier = classifier
        self.__parameters_mapper : dict = parameters_mapper
        self.__gene_space : dict = gene_space
        self.__extra_GA_parameters : dict = extra_GA_parameters
        self.__results : list[dict[str:Any]] = []
    
    def run_all(self) -> None:
        """
        In the foldername we expect : 
            foldername
                L XXX
                    L checkpoint-XXX
                        L ...
                        L training_args.bin
                    L data
                    L embeddings
                        L epoch_XXX
                            L train_embeddings.pt
                            L train_labels.pt
                            L test_embeddings.pt
                            L test_labels.pt
                L XXX
                    L ...
        """
        # UPGRADE add some security
        all_posible_folders : list[str] = os.listdir(f"{ROOT_MODELS}/{self.__foldername}")
        all_possible_parameters : list[dict[str:Any]] = []
        for folder in all_posible_folders : 
            all_checkpoints : list[str] = os.listdir(f"{ROOT_MODELS}/{self.__foldername}/{folder}") # One checkpoint per epoch
            first_checkpoint : str = all_checkpoints[0]
            training_args = load((f"{ROOT_MODELS}/{self.__foldername}/{folder}/"
                                  f"{first_checkpoint}/training_args.bin"),
                                  weights_only=False)
            # add one row per epoch
            for epoch in range(1, len(all_posible_folders) + 1):
                all_possible_parameters.append({
                    "learning_rate" : training_args.learning_rate,
                    "optim" : training_args.optim,
                    "warmup_ratio" : training_args.warmup_ratio,
                    "weight_decay" : training_args.weight_decay,
                    "path" : f"{ROOT_MODELS}/{self.__foldername}/{folder}/embeddings/epoch_{epoch}",
                    "epoch" : epoch
                })
        all_possible_parameters : pd.DataFrame = pd.DataFrame(all_possible_parameters)
        print(all_possible_parameters)
        # TODO implement different iterations
        for x in product(*self.__ranges_of_parameters.values()) :
            condition = True
            for i, key in enumerate(self.__ranges_of_parameters) : 
                condition = (condition) & (all_possible_parameters[key] == x[i])
            # TODO rename the variables
            corresponding_path : pd.Series = \
                all_possible_parameters.loc[condition, :]
            if len(corresponding_path) > 0 : 
                path = corresponding_path.iloc[0]["path"]
                print(f"{x} : {path}")

                data = DataHandlerForGOfSC(path) # TODO implement n_sample

                optimiser = GeneticOptimiserForSklearnClassifier(
                    data = data, 
                    classifier = self.__classifier, 
                    parameters_mapper = self.__parameters_mapper, 
                    gene_space = self.__gene_space, 
                    extra_GA_parameters = self.__extra_GA_parameters
                )
                optimum, f1_max, optimisation_time, n_optim_iterations = optimiser.run()
                self.__results.append({
                    **optimum,
                    "f1_macro" : f1_max, 
                    "time" : optimisation_time,
                    "n_optim_iterations" : n_optim_iterations,
                    "learning_rate" : corresponding_path.iloc[0]["learning_rate"],
                    "optim" : corresponding_path.iloc[0]["optim"],
                    "warmup_ratio" : corresponding_path.iloc[0]["warmup_ratio"],
                    "weight_decay" : corresponding_path.iloc[0]["weight_decay"],
                    "path" : corresponding_path.iloc[0]["path"],
                    "epoch" : corresponding_path.iloc[0]["epoch"],
                    "classifier" : self.__classifier.__name__
                })

                del optimum, f1_max, optimisation_time, n_optim_iterations, optimiser, data
                clean()
                
            else :
                print(f"{x} : PASS")
                pass            
    def save_results(self, filename : str) -> None: 
        """
        """
        if len(self.__results) > 0 : 
            try : 
                # if file already exists
                df = pd.read_csv(f"{ROOT_RESULTS}/{filename}")
                df = pd.concat((df, pd.DataFrame(self.__results)))
            except:
                df = pd.DataFrame(self.__results)
            finally:
                df.to_csv(f"{ROOT_RESULTS}/{filename}", index = False)
            

    def routine(self, filename : str) -> None:
        """
        """
        self.run_all()
        self.save_results(filename)

        

