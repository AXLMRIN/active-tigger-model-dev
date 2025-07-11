# IMPORTS ######################################################################
import pandas as pd
from mergedeep import merge
from typing import Union
from . import LAYOUT
from .figure_tools import multiple_figures_layout, generic_bar, generic_scatter_with_bands
from ..general import SUL_string, get_band, auto_log_range, get_uniques_values, get_most_frequent_item
from plotly.graph_objs._figure import Figure
# CONSTANTS ####################################################################
PLOTLY_SAVING_KWARGS = {
    "full_html" : False,
    "auto_play" : False,
    "include_plotlyjs" : False, 
    "include_mathjax" : False,
    "config" : {"responsive" : True} # Not sure it works nor it's useful
}
# SCRIPTS ######################################################################
class Visualisation :
    """
    """
    def __init__(self,
        data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame,
        type : str,
        column_frame : str = "embedding_model", 
        column_trace : str = "classifier", 
        score_column : str = "score",
        x_axis_column : str|None = None,
        measure : str = "f1_macro",
        column_measure : str = "measure") -> None:
        """
        """
        self.__data_baseline : pd.DataFrame = data_baseline
        self.__data_others : pd.DataFrame = data_others
        self.__type : str = type # UPGRADE add verifications bar can't exist with x_axis and 
        self.__measure : str = measure
        self.__column_frame : str = column_frame
        self.__column_trace : str = column_trace
        self.__column_score : str = score_column
        self.__column_measure : str = column_measure
        self.__column_x_axis : str = x_axis_column
        self.__fig : Figure = Figure(layout = LAYOUT) # LAYOUT is the general theme``
        
        (self.__list_of_frames, self.__list_of_traces) = (None,) * 2

    def preprocess_data(self, alpha : float = 0.9) -> None:
        """
        """
        # Select columns
        columns_to_retrieve = [col for col in [self.__column_frame,
            self.__column_trace, self.__column_score, self.__column_measure, 
            self.__column_x_axis] if col is not None]
        self.__data_baseline = self.__data_baseline.\
            loc[:,columns_to_retrieve]
        self.__data_others = self.__data_others.\
            loc[:,columns_to_retrieve]
        
        # Select rows where the measure matches
        measure_condition = \
            self.__data_baseline[self.__column_measure] == self.__measure
        self.__data_baseline = self.__data_baseline.loc[measure_condition, :]
        measure_condition = \
            self.__data_others[self.__column_measure] == self.__measure
        self.__data_others = self.__data_others.loc[measure_condition, :]
        
        self.__list_of_frames = SUL_string([
            *self.__data_baseline[self.__column_frame], 
            *self.__data_others[self.__column_frame]
        ])

        self.__list_of_traces = SUL_string([
            *self.__data_baseline[self.__column_trace], 
            *self.__data_others[self.__column_trace]
        ])

        # Compute Mean and Condidence Intervals of for the score
        get_lower_band = lambda col : get_band(col, "lower", alpha)
        get_upper_band = lambda col : get_band(col, "upper", alpha)
        columns_to_groupby : list[str] = [col 
            for col in [self.__column_frame, self.__column_trace, self.__column_x_axis]
            if col is not None
        ]
        self.__data_baseline_M_and_CI = self.__data_baseline.\
            groupby(columns_to_groupby, as_index = False).\
            agg(
                mean = (self.__column_score, "mean"),
                lower_band = (self.__column_score, get_lower_band),
                upper_band = (self.__column_score, get_upper_band)
            )

        self.__data_others_M_and_CI = self.__data_others.\
            groupby(columns_to_groupby, as_index = False).\
            agg(
                mean = (self.__column_score, "mean"),
                lower_band = (self.__column_score, get_lower_band),
                upper_band = (self.__column_score, get_upper_band)
            )
        
    def __add_bar(self) -> None: 
        """
        """
        grouped_baseline = self.__data_baseline_M_and_CI.\
            groupby([self.__column_frame, self.__column_trace],as_index = False)
        for (frame,trace), sub_df in grouped_baseline: 
            idx = self.__list_of_frames.index(frame)
            bars = generic_bar(
                df = sub_df, 
                col_x = self.__column_trace, 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = trace, 
                idx = idx
            )
            self.__fig.add_trace(bars)

        # Create the bars for the other classifiers
        grouped_others = self.__data_others_M_and_CI.\
            groupby([self.__column_frame, self.__column_trace], as_index=False)
        for (frame, trace), sub_df in grouped_others:
            idx = self.__list_of_frames.index(frame)
            bars = generic_bar(
                df = sub_df, 
                col_x = self.__column_trace, 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = trace,
                idx = idx
            )
            self.__fig.add_trace(bars)
    
    def __add_scatter(self) -> None:
        """
        """
        # Create the scatter for the baseline
        grouped_baseline = self.__data_baseline_M_and_CI.\
            groupby([self.__column_frame, self.__column_trace],as_index = False)
        for (frame,trace), sub_df in grouped_baseline: 
            idx = self.__list_of_frames.index(frame)
            trace_mean, trace_bands = generic_scatter_with_bands(
                df = sub_df, 
                col_x = self.__column_x_axis, 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = trace,
                idx = idx
            )
            
            self.__fig.add_trace(trace_mean)
            self.__fig.add_trace(trace_bands)

        # Create the scatter for the other classifiers
        grouped_others = self.__data_others_M_and_CI.\
            groupby([self.__column_frame, self.__column_trace], as_index = False)
        for (frame, trace), sub_df in grouped_others:
            idx = self.__list_of_frames.index(frame)
            trace_mean, trace_bands = generic_scatter_with_bands(
                df = sub_df, 
                col_x = self.__column_x_axis, 
                col_y = "mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = trace, 
                idx = idx
            )
            self.__fig.add_trace(trace_mean)
            self.__fig.add_trace(trace_bands)

    def build_figure(self, additional_xaxis_kwargs : dict = {},
        figure_layout_kwargs : dict = {}) -> None : 
        """
        """
        multiple_figures_layout(
            self.__fig, 
            self.__list_of_frames, 
            xaxis_kwargs = merge(
                {'categoryorder' : "trace", 'type' : "category"},
                additional_xaxis_kwargs),
            y_label = self.__measure
        )
        self.__fig.update_layout(figure_layout_kwargs)
        
        # Create the bars for the baseline
        if self.__type == "bar" : 
            self.__add_bar()
        elif self.__type == "scatter" : 
            self.__add_scatter()
        else:
            raise(TypeError,"type must be equal to 'bar' or 'scatter'")
    
    def return_figure(self) -> Figure:
        """
        """
        return self.__fig
    
    def save(self, filename : str) -> None : 
        """
        """
        self.__fig.write_image(filename)

    def routine(self, alpha : float = 0.9, additional_xaxis_kwargs : dict = {}, 
                figure_layout_kwargs : dict = {}) -> Figure:
        self.preprocess_data(alpha)
        self.build_figure(figure_layout_kwargs)
    
def plot_score_per_embedding_model_and_classifier(data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame, filename : str|None = None, 
        return_html : bool = False, return_fig : bool = False) -> Figure|None:
    """
    """
    visu = Visualisation(
        data_baseline = data_baseline,
        data_others = data_others,
        type = "bar",
        column_frame = "embedding_model",
        column_trace = "classifier",
        score_column = "score",
        x_axis_column = None,
        measure = "f1_macro",
        column_measure = "measure"
    )
    visu.routine()
    if return_fig : 
        return visu.return_figure()
    if return_html: 
        return visu.return_figure().to_html(**PLOTLY_SAVING_KWARGS)
    if filename is not None : 
        visu.save(filename)



def plot_score_per_classifier_and_embedding_model(data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame, filename : str|None = None, 
        return_html : bool = False, return_fig : bool = False) -> Figure|None:
    """
    """
    visu = Visualisation(
        data_baseline = data_baseline,
        data_others = data_others,
        type = "bar",
        column_frame = "classifier",
        column_trace = "embedding_model",
        score_column = "score",
        x_axis_column = None,
        measure = "f1_macro",
        column_measure = "measure"
    )
    visu.routine()
    if return_fig : 
        return visu.return_figure()
    if return_html: 
        return visu.return_figure().to_html(**PLOTLY_SAVING_KWARGS)
    if filename is not None : 
        visu.save(filename)

def plot_score_against_learning_rate_per_embedding_model_and_classifier(
    data_baseline : pd.DataFrame, data_others : pd.DataFrame,
    filename : str|None = None, return_html : bool = False, 
    return_fig : bool = False) -> Figure|None:
    """
    """
    visu = Visualisation(
        data_baseline = data_baseline,
        data_others = data_others,
        type = "scatter",
        column_frame = "embedding_model",
        column_trace = "classifier",
        score_column = "score",
        x_axis_column = "learning_rate",
        measure = "f1_macro",
        column_measure = "measure"
    )
    visu.routine()
    if return_fig : 
        return visu.return_figure()
    if return_html: 
        return visu.return_figure().to_html(**PLOTLY_SAVING_KWARGS)
    if filename is not None : 
        visu.save(filename)