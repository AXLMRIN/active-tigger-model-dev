# IMPORTS ######################################################################
import os
import pandas as pd
from .Visualisation import (plot_score_per_embedding_model_and_classifier,
    plot_score_per_classifier_and_embedding_model,
    plot_score_against_learning_rate_per_embedding_model_and_classifier)
from .Table import (table_score_against_epoch_per_classifier_and_embedding_model)
from jinja2 import Template
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
        self.__figures : dict[str:str] = {}

    def open_data(self) : 
        """
        """
        self.__baseline : pd.DataFrame = \
            pd.read_csv(f"{self.__filename_baseline}")
        self.__others : pd.DataFrame = \
            pd.read_csv(f"{self.__filename_others}")

    def create_and_save_figures(self) -> None : 
        """
        """
        input = {
            "data_baseline" : self.__baseline,
            "data_others" : self.__others,
            "return_html" : True
        }

        self.__figures["Score par classifieur"] = \
            plot_score_per_embedding_model_and_classifier(**input)
        self.__figures["Score par modèle de plongement"] = \
            plot_score_per_classifier_and_embedding_model(**input)
        self.__figures["Score par Learning Rate"] = \
            plot_score_against_learning_rate_per_embedding_model_and_classifier(**input)
        self.__figures["Table du score par nombre d'epochs"] = \
            table_score_against_epoch_per_classifier_and_embedding_model(**input)
    
    def export_to_hmtl(self, main_title : str, foldername : str) -> None:
        """
        """
        if not(os.path.exists(foldername)): os.makedirs(foldername)

        jinja_data = {"main_title" : main_title}
        for i,key in enumerate(self.__figures):
            jinja_data[f"title_{i}"] = key
            jinja_data[f"fig_{i}"] = self.__figures[key]

        with open(f"{foldername}/index.html", "w", encoding="utf-8") as output_file:
            with open("./src/toolbox/visualise_results/template.html") as template_file:
                j2_template = Template(template_file.read())
                output_file.write(j2_template.render(jinja_data))
            
    def routine(self, main_title : str, foldername : str) -> None : 
        """
        """
        self.open_data()
        print(self.__baseline) #TODELETE
        print(self.__others) #TODELETE
        self.create_and_save_figures()
        self.export_to_hmtl(main_title, foldername)