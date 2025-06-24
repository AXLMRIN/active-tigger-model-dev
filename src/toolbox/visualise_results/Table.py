# IMPORTS ######################################################################
import pandas as pd
import numpy as np
from ..general import SUL_string, get_band, pretty_mean_and_ci
from plotly.graph_objs._figure import Figure
import plotly.graph_objects as go
from great_tables import GT, style, loc
from great_tables.gt import GT as gt_table
# CONSTANT #####################################################################
LIGHT_GREY = "#F0F0F0"
# SCRIPTS ######################################################################
class Table : 
    """
    """
    def __init__(self, 
        data_baseline : pd.DataFrame, 
        data_others : pd.DataFrame, 
        column_row : str = "classifier", 
        column_column : str = "epoch",
        column_group : str = "embedding_model",
        column_score : str = "score", 
        measure : str = "f1_macro", 
        column_measure : str = "measure") -> None:
        """
        """
        self.__data_baseline : pd.DataFrame = data_baseline
        self.__data_others : pd.DataFrame = data_others
        self.__column_row : str = column_row
        self.__column_column : str = column_column
        self.__column_group : str = column_group
        self.__column_score : str = column_score
        self.__column_measure : str = column_measure
        self.__measure : str = measure
        self.__fig = None

    def preprocess(self, alpha : float = 0.9) -> None:
        """
        """
        # Select columns
        columns_to_retrieve = [col for col in [self.__column_row,
        self.__column_column, self.__column_group, self.__column_score, 
            self.__column_measure] if col is not None]
        
        self.__data_baseline = self.__data_baseline.\
            loc[:,columns_to_retrieve]
        self.__data_others = self.__data_others.\
            loc[:,columns_to_retrieve]
        # There is no need to keep the dataset appart, so we are merging them from now
        data = pd.concat((self.__data_baseline, self.__data_others))

        # Select rows where the measure matches
        measure_condition = data[self.__column_measure] == self.__measure
        data = data.loc[measure_condition, :]

        # Compute Mean and Condidence Intervals of for the score
        get_lower_band = lambda col : get_band(col, "lower", alpha)
        get_upper_band = lambda col : get_band(col, "upper", alpha)
        columns_to_groupby : list[str] = [col 
            for col in [self.__column_group, self.__column_row, self.__column_column]
            if col is not None
        ]
        data_M_and_CI = data.\
            groupby(columns_to_groupby, as_index = False).\
            agg(
                mean = (self.__column_score, "mean"),
                lower_band = (self.__column_score, get_lower_band),
                upper_band = (self.__column_score, get_upper_band)
            )
        data_M_and_CI["text"] = data_M_and_CI.\
            apply(pretty_mean_and_ci, axis = 1)
        
        # Drop unnecessary columns
        data_M_and_CI = data_M_and_CI.\
            loc[:,[self.__column_group, self.__column_column, self.__column_row,
                "text"]]
        
        # Only work with strings
        for column in data_M_and_CI : 
            #NOTE Raises an error that makes no sense
            data_M_and_CI.loc[:,column] = data_M_and_CI.loc[:,column].astype(str)
        
        self.__list_of_columns = SUL_string(data_M_and_CI[self.__column_column])

        data_table_format = []
        data_M_and_CI_grouped = data_M_and_CI.groupby([self.__column_group, self.__column_row], as_index = False)
        for (group, row), sub_df in data_M_and_CI_grouped:
            table_row = {
                self.__column_group : group,
                self.__column_row : row
            }
            for column in self.__list_of_columns : 
                # NOTE some data may not exist for all column so we need to 
                # implement a try / error catching
                try : 
                    table_row[column] =  sub_df.\
                        loc[sub_df[self.__column_column] == column,"text"].item()
                except:
                    table_row[column] = "NaN±NaN"
            
            data_table_format.append(table_row)
            
        self.__data_table_format = pd.DataFrame(data_table_format)
    
    def build_table(self) -> None:
        """
        """
        best_results : dict[str:list[int]] = {}
        for col in self.__list_of_columns : 
            best_results[col] = []
            for row in range(len(self.__data_table_format)):
                value_cell = self.__data_table_format.iloc[row][col]
                value_cell = float(value_cell.split("±")[0]) - float(value_cell.split("±")[1])
                values_row = self.__data_table_format.iloc[row][self.__list_of_columns]
                values_row = [float(value.split("±")[0]) - float(value.split("±")[1]) 
                              for value in values_row]
                max_values_row = max(values_row)
                if  value_cell == max_values_row:
                    best_results[col].append(row)

        self.__fig = (
            GT(self.__data_table_format)
            .tab_stub(
                rowname_col=self.__column_row,
                groupname_col=self.__column_group
            )
            .tab_stubhead(
                label = self.__column_row
            )
            .tab_spanner(
                label = self.__column_column, 
                columns = self.__list_of_columns
            )
            .tab_style(
                style = style.text(align = "center"),
                locations = loc.column_header()
            )
            .opt_horizontal_padding(scale = 3)
            .opt_vertical_padding(scale = 1.5)
            .tab_style(
                style = style.text(color="Navy", weight = "bold"),
                locations= [
                    loc.body(columns=[col], rows = best_results[col])
                    for col in best_results
                ]
            )
            .tab_style(
                style = [style.fill(color=LIGHT_GREY), style.text(weight = "bold",size="large")],
                locations= loc.row_groups()
            )

        )
    
    def return_fig(self) -> gt_table: 
        """
        """
        return self.__fig

    def save(self, filename : str) -> None : 
        """
        """
        self.__fig.save(filename)

    def routine(self, alpha : float = 0.9) -> gt_table:
        self.preprocess(alpha)
        self.build_table()
        return self.__fig
    
def table_score_against_epoch_per_classifier_and_embedding_model(
    data_baseline : pd.DataFrame, data_others : pd.DataFrame, 
    filename : str|None = None, return_figure : bool = False,
    return_html : bool = False) -> gt_table|None:
    """
    """
    t = Table(
        data_baseline=data_baseline, 
        data_others = data_others,
        column_row = "classifier",
        column_column= "epoch",
        column_group = "embedding_model",
        column_score = "score",
        measure = "f1_macro",
        column_measure = "measure"
    )
    t.routine()
    if return_figure : 
        return t.return_fig()
    if return_html:
        return t.return_fig().as_raw_html()
    if filename is not None:
        t.save(filename)

def table_score_against_learning_rate_per_classifier_and_embedding_model(
    data_baseline : pd.DataFrame, data_others : pd.DataFrame, 
    filename : str|None = None, return_figure : bool = False,
    return_html : bool = False) -> gt_table|None:
    """
    """
    t = Table(
        data_baseline=data_baseline, 
        data_others = data_others,
        column_row = "classifier",
        column_column= "learning_rate",
        column_group = "embedding_model",
        column_score = "score",
        measure = "f1_macro",
        column_measure = "measure"
    )
    t.routine()
    if return_figure : 
        return t.return_fig()
    if return_html:
        return t.return_fig().as_raw_html()
    if filename is not None:
        t.save(filename)