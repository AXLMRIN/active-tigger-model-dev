# IMPORTS ######################################################################
import pandas as pd
from .Visualisation import (plot_score_per_embedding_model_and_classifier,
    plot_score_per_classifier_and_embedding_model,
    plot_score_against_learning_rate_per_embedding_model_and_classifier)
from .Table import (table_score_against_epoch_per_classifier_and_embedding_model)
# SCRIPTS ######################################################################
class VisualiseAll : 
    """
    """
    def __init__(self, 
        filename_baseline : str, 
        filename_others : str) -> None:
        """
        """
        self.__filename_baseline : str = filename_baseline
        self.__filename_others : str = filename_others

        (self.__baseline, self.__others) = (None, ) * 2

    def open_data(self) : 
        self.__baseline : pd.DataFrame = \
            pd.read_csv(f"{self.__filename_baseline}")
        self.__others : pd.DataFrame = \
            pd.read_csv(f"{self.__filename_others}")

    def create_figures(self) -> None : 
        input = {
            "data_baseline" : self.__baseline,
            "data_others" : self.__others
        }
        plot_score_per_embedding_model_and_classifier(**input)
        plot_score_per_classifier_and_embedding_model(**input)
        plot_score_against_learning_rate_per_embedding_model_and_classifier(**input)
        table_score_against_epoch_per_classifier_and_embedding_model(**input)
        
    def routine(self) -> None : 
        self.open_data()
        print(self.__baseline) #TODELETE
        print(self.__others) #TODELETE
        self.create_figures()