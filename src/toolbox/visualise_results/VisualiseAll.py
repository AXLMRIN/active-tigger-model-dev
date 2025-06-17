# IMPORTS ######################################################################
import pandas as pd
from .. import ROOT_RESULTS
from .FigureObjects import ScorePerModelAndClassifier, ScorePerLearningRateAndModelAndClassifier
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
            pd.read_csv(f"{ROOT_RESULTS}/{self.__filename_baseline}")
        self.__others : pd.DataFrame = \
            pd.read_csv(f"{ROOT_RESULTS}/{self.__filename_others}")

    def create_figures(self) -> None : 
        ScorePerModelAndClassifier(self.__baseline, self.__others).\
            routine().\
            show()
        ScorePerLearningRateAndModelAndClassifier(self.__baseline, self.__others).\
            routine().\
            show()
        
    def routine(self) -> None : 
        self.open_data()
        print(self.__baseline) #TODELTE
        print(self.__others) #TODELETE
        self.create_figures()