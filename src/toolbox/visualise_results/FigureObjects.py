# IMPORTS ######################################################################
import pandas as pd
from .. import ROOT_RESULTS
from . import LAYOUT
from .figure_tools import multiple_figures_layout, generic_bar, generic_scatter_with_bands
from ..general import SUL_string, get_band
from plotly.graph_objs._figure import Figure
# SCRIPTS ######################################################################
class ScorePerModelAndClassifier:
    """
    """
    def __init__(self, 
        data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame,
        measure : str = "f1_macro") -> None:
        """
        """
        self.__data_baseline : pd.DataFrame = data_baseline
        self.__data_others : pd.DataFrame = data_others
        self.__measure : str = measure
        self.__fig = Figure(layout = LAYOUT) # LAYOUT is the general theme
    
    def preprocess_data(self, alpha : float = 0.9) -> None:
        """
        """
        # Select columns
        self.__data_baseline = self.__data_baseline.\
            loc[:,["embedding_model","score", "measure"]]
        # Add a column "Classifier" : 
        self.__data_baseline["classifier"] = ["Baseline - HF Classifier"] * \
                                                len(self.__data_baseline)
        self.__data_others = self.__data_others.\
            loc[:,["embedding_model","classifier","score", "measure"]]
        
        # Select rows where the
        measure_condition = self.__data_baseline["measure"] == self.__measure
        self.__data_baseline = self.__data_baseline.loc[measure_condition, :]
        measure_condition = self.__data_others["measure"] == self.__measure
        self.__data_others = self.__data_others.loc[measure_condition, :]
        
        self.__list_of_embedding_models = SUL_string([
            *self.__data_baseline["embedding_model"], 
            *self.__data_others["embedding_model"]
        ])

        self.__list_of_classifiers = SUL_string([
            "Baseline - HF Classifier", 
            *self.__data_others["classifier"]
        ])

        self.__data_baseline_M_and_CI = self.__data_baseline.\
            groupby(["embedding_model", "classifier"], as_index = False).\
            agg(
                mean = ("score", "mean"),
                lower_band = ("score", lambda col : get_band(col, "lower", alpha)),
                upper_band = ("score", lambda col : get_band(col, "upper", alpha))
            )

        self.__data_others_M_and_CI = self.__data_others.\
            groupby(["embedding_model", "classifier"], as_index = False).\
            agg(
                mean = ("score", "mean"),
                lower_band = ("score", lambda col : get_band(col, "lower", alpha)),
                upper_band = ("score", lambda col : get_band(col, "upper", alpha))
            )

    def build_figure(self, figure_layout_kwargs : dict = {}) -> None : 
        """
        """
        multiple_figures_layout(
            self.__fig, 
            self.__list_of_embedding_models, 
            xaxis_kwargs = {'categoryorder' : "trace", 'type' : "category"},
            y_label = self.__measure
        )
        self.__fig.update_layout(figure_layout_kwargs)
        
        # Create the bars for the baseline
        grouped_baseline = self.__data_baseline_M_and_CI.groupby(["embedding_model", "classifier"])
        for idx, embedding_model in enumerate(self.__list_of_embedding_models): 
            sub_df = grouped_baseline.get_group((embedding_model, "Baseline - HF Classifier"))
            bars = generic_bar(
                df = sub_df, 
                col_x = "classifier", 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = "Baseline - HF Classifier", 
                idx = idx
            )
            self.__fig.add_trace(bars)

        # Create the bars for the other classifiers
        grouped_others = self.__data_others_M_and_CI.groupby(["embedding_model", "classifier"])
        for idx, embedding_model in enumerate(self.__list_of_embedding_models): 
            for classifer in self.__list_of_classifiers:
                # NOTE some classifiers are not tested accross all empedding models
                # ex : 'Baseline - HF Classifier'
                try : 
                    sub_df = grouped_others.get_group((embedding_model, classifer))
                    bars = generic_bar(
                        df = sub_df, 
                        col_x = "classifier", 
                        col_y = "mean",
                        col_band_u = "upper_band", 
                        col_band_l = "lower_band",
                        name = classifer,
                        idx = idx
                    )
                    self.__fig.add_trace(bars)
                except : 
                    pass

    def routine(self, alpha : float = 0.9, figure_layout_kwargs : dict = {}) -> Figure:
        self.preprocess_data(alpha)
        self.build_figure(figure_layout_kwargs)
        return self.__fig

class ScorePerLearningRateAndModelAndClassifier:
    """
    """
    def __init__(self, 
        data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame,
        measure : str = "f1_macro") -> None:
        """
        """
        self.__data_baseline : pd.DataFrame = data_baseline
        self.__data_others : pd.DataFrame = data_others
        self.__measure : str = measure
        self.__fig = Figure(layout = LAYOUT) # LAYOUT is the general theme
    
    def preprocess_data(self, alpha : float = 0.9) -> None:
        """
        """
        # Select columns
        self.__data_baseline = self.__data_baseline.\
            loc[:,["embedding_model", "learning_rate","score", "measure"]]
        # Add a column "Classifier" : 
        self.__data_baseline["classifier"] = ["Baseline - HF Classifier"] * \
                                                len(self.__data_baseline)
        self.__data_others = self.__data_others.\
            loc[:,["embedding_model","classifier","learning_rate", "score", "measure"]]
        
        # Select rows where the
        measure_condition = self.__data_baseline["measure"] == self.__measure
        self.__data_baseline = self.__data_baseline.loc[measure_condition, :]
        measure_condition = self.__data_others["measure"] == self.__measure
        self.__data_others = self.__data_others.loc[measure_condition, :]
        
        self.__list_of_embedding_models = SUL_string([
            *self.__data_baseline["embedding_model"], 
            *self.__data_others["embedding_model"]
        ])

        self.__list_of_classifiers = SUL_string([
            "Baseline - HF Classifier", 
            *self.__data_others["classifier"]
        ])

        self.__data_baseline_M_and_CI = self.__data_baseline.\
            groupby(["embedding_model", "classifier", "learning_rate"], as_index = False).\
            agg(
                mean = ("score", "mean"),
                lower_band = ("score", lambda col : get_band(col, "lower", alpha)),
                upper_band = ("score", lambda col : get_band(col, "upper", alpha))
            )

        self.__data_others_M_and_CI = self.__data_others.\
            groupby(["embedding_model", "classifier", "learning_rate"], as_index = False).\
            agg(
                mean = ("score", "mean"),
                lower_band = ("score", lambda col : get_band(col, "lower", alpha)),
                upper_band = ("score", lambda col : get_band(col, "upper", alpha))
            )

    def build_figure(self, figure_layout_kwargs : dict = {}) -> None : 
        """
        """
        multiple_figures_layout(
            self.__fig, 
            self.__list_of_embedding_models, 
            xaxis_kwargs = {
                "tickvals" : get_uniques_values(
                    self.__data_baseline["learning_rate"],
                    self.__data_others["learning_rate"]),
                'range' : auto_log_range(self.__data_baseline["learning_rate"],
                                          self.__data_others["learning_rate"]),
                'type' : "log"},
            y_label = self.__measure,
            xlabel_prefix = "Learning rate<br><br>"
        )
        self.__fig.update_layout(figure_layout_kwargs)
        
        # Create the bars for the baseline
        grouped_baseline = self.__data_baseline_M_and_CI.groupby(["embedding_model", "classifier"])
        for idx, embedding_model in enumerate(self.__list_of_embedding_models): 
            sub_df = grouped_baseline.get_group((embedding_model, "Baseline - HF Classifier"))
            trace_mean, trace_bands = generic_scatter_with_bands(
                df = sub_df, 
                col_x = "learning_rate", 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = "Baseline - HF Classifier", 
                idx = idx
            )
            
            self.__fig.add_trace(trace_mean)
            self.__fig.add_trace(trace_bands)

        # Create the bars for the other classifiers
        grouped_others = self.__data_others_M_and_CI.groupby(["embedding_model", "classifier"])
        for idx, embedding_model in enumerate(self.__list_of_embedding_models): 
            for classifer in self.__list_of_classifiers:
                # NOTE some classifiers are not tested accross all empedding models
                # ex : 'Baseline - HF Classifier'
                try : 
                    sub_df = grouped_others.get_group((embedding_model, classifer))
                    trace_mean, trace_bands = generic_scatter_with_bands(
                        df = sub_df, 
                        col_x = "learning_rate", 
                        col_y = "mean",
                        col_band_u = "upper_band", 
                        col_band_l = "lower_band",
                        name = classifer, 
                        idx = idx
                    )

                    self.__fig.add_trace(trace_mean)
                    self.__fig.add_trace(trace_bands)
                except : 
                    pass

    def routine(self, alpha : float = 0.9, figure_layout_kwargs : dict = {}) -> Figure:
        self.preprocess_data(alpha)
        self.build_figure(figure_layout_kwargs)
        return self.__fig
