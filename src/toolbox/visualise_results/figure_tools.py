# IMPORTS ######################################################################
import pandas as pd
from .. import ROOT_RESULTS
from . import GRIDCOLOR_X, COLORS
from ..general import SUL_string
from plotly.graph_objs._figure import Figure
import plotly.graph_objects as go
# SCRIPTS ######################################################################
def multiple_figures_layout(
        fig : Figure, 
        list_of_frames_names : list[str],  
        xaxis_kwargs : dict = {},   
        y_label : str = "Y Label",
        xlabel_prefix : str = ""
    ):
    """
    """
    n_frames : int = len(list_of_frames_names)
    if n_frames >1:
        subplot_width =  0.9 / n_frames
        gap = 0.1 / (n_frames - 1)
        fig.update_layout({
            'xaxis'  : {
                'anchor' : "x1" , 
                'domain' : [0.0,subplot_width],
                'title' : {'text' : f"{xlabel_prefix} {list_of_frames_names[0]}"},
                'gridcolor' : GRIDCOLOR_X,
                "zerolinecolor" : GRIDCOLOR_X,
                **xaxis_kwargs
            },
            "yaxis_title_text" : y_label,
            "yaxis_range" : [0,1],
            **{
                f'xaxis{i+1}' : {
                    'anchor' : f"x{i+1}" , 
                    'domain' : [
                        (subplot_width + gap) * i, 
                        min((subplot_width + gap) * i + subplot_width, 1)
                        ],
                    'title' : {'text' : f"{xlabel_prefix} {list_of_frames_names[i]}"},
                    'gridcolor' : GRIDCOLOR_X,
                    "zerolinecolor" : GRIDCOLOR_X,
                    **xaxis_kwargs
                }
                for i in range(1, n_frames)
            }
        })
    else : 
        fig.update_layout({
            'xaxis'  : {
                'anchor' : "x1" , 
                'domain' : [0.0,1],
                'title' : {'text' : list_of_frames_names[0]},
                'gridcolor' : GRIDCOLOR_X,
                "zerolinecolor" : GRIDCOLOR_X,
                **xaxis_kwargs
            },
            "yaxis_title_text" : y_label,
            "yaxis_range" : [0,1]
        })

def generic_bar(
        df : pd.DataFrame, 
        col_x : str, 
        col_y : str, 
        col_band_u : str, 
        col_band_l : str,
        name : str, 
        idx : int
    ):
    """
    """
    return go.Bar(
        x = df[col_x],
        y = df[col_y],
        error_y = {
            'type' : "data", 
            'symmetric' : False,
            'array' : df[col_band_u],
            'arrayminus' : df[col_band_l],
            'color' : COLORS["error"],
            'width' : 20,
            'thickness' : 2
        },
        name = name,
        marker = {
            'color' : COLORS[name],
            'cornerradius' : 5
        },
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = (idx == 0)
    )

def error_band_color(rgb_color, error_band_opacity = 0.2):
    """
    """
    out = rgb_color[:3] + "a" + rgb_color[3:-1] + f",{error_band_opacity})"
    return out

def generic_scatter_with_bands(df, col_x, col_y, col_band_u, col_band_l, name, idx = 0):
    """
    """
    trace_1 = go.Scatter(
        x = df[col_x],
        y = df[col_y],
        name = name,
        marker = {
            'color' : COLORS[name],
        },
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = (idx == 0),
        zorder = 1
    )

    trace_2 = go.Scatter(
        x = [*df[col_x],*df[col_x][::-1]],
        y = [*df[col_band_u], *df[col_band_l][::-1]],
        name = name,
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = True,
        #Filling
        fill ='toself',
        fillcolor = error_band_color(COLORS[name]),
        line = dict(color='rgba(0,0,0,1)'),
        hoverinfo = "skip",
        zorder = 0
    )
    return trace_1, trace_2