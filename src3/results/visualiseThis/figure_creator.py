# === === === === === === === === === === === === === === === === === === === ==
# IMPORTS
# === === === === === === === === === === === === === === === === === === === ==
from .data_handling import get_mean_and_half_band
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
import pandas as pd
import numpy as np

# === === === === === === === === === === === === === === === === === === === ==
# LOCAL VARIABLES
# === === === === === === === === === === === === === === === === === === === ==
layout_general_parameters = {
    'width' : 1200, 'height' : 600,
    'paper_bgcolor' : "white",
    'plot_bgcolor' : "rgba(189, 224, 254,0.15)",
    'yaxis' : {
        'gridcolor' : "rgb(34,34,34)"
    },
    "legend" : {
        "orientation" : "h",
        "x" : 0.5, "xanchor" : "center",
        "y" : 1.01, "yanchor" : "bottom"
    }
}
gridcolor_x = "rgba(100,100,100,0.5)"

colors = {
    "KNN"                    : "rgb(230,  57,  71)", 
    "Random Forest"          : "rgb(255, 183,   3)", 
    "MLPClassifier (slkearn)" : "rgb( 69, 123, 157)", 
    "MLPClassifier (HF)"      : "rgb(144, 190, 109)", 
    
    'answerdotai/ModernBERT-base'   : "rgb(255,   0,  85)", 
    'FacebookAI/roberta-base'       : "rgb(255,  85,   0)", 
    'google-bert/bert-base-uncased' : "rgb( 56,   0, 153)",

    500                         : "rgb(144, 225, 239)",
    1000                        : "rgb(  0, 180, 216)",
    1500                        : "rgb(  2,  61, 138)",
    2000                        : "rgb(  0,  53,  84)",
    "Entraînement Hugging Face" : "rgb(144, 190, 109)", 

    "error" : "#000000",
    "marker" : "#6b705c",
}

meilleurs_models = {
    "Few shot" : 0.577,
    "Zero shot" : 0.6,
    "Base line finetuned" : 0.648
}

dash_meilleurs_models = {
    "Few shot" : "dot",
    "Zero shot" : "dashdot",
    "Base line finetuned" : "longdash"
}

# === === === === === === === === === === === === === === === === === === === ==
# FUNCTIONS
# === === === === === === === === === === === === === === === === === === === ==
def f1_macro_per_model_and_method(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "method", "f1_macro"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model","method"],
        column = "f1_macro",
        N = N
    )
    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    nModels = len(listOfModels)

    multiple_figures_layout(fig, nModels,listOfModels, 
        xaxis_kwargs = {'categoryorder' : "trace", 'type' : "category"})

    # Create bars for each model and method
    to_print["upper_band"] = (to_print["CI_f1_macro_upper"] - 
                              to_print["f1_macro_mean"])
    to_print["lower_band"] = -(to_print["CI_f1_macro_lower"] - 
                              to_print["f1_macro_mean"])
    
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))
            fig.add_trace(generic_bar(
                sub_df = sub_df, 
                col_x = "method", 
                col_y = "f1_macro_mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = method, 
                idx = idx
            ))

    # Create markers for each model and method
    grouped = selected_data.groupby(["model","method"])
    showlegend = True
    for idx, model in enumerate(listOfModels):  
        for method in listOfMethods:
            sub_df = grouped.get_group((model,method))
            y = sub_df["f1_macro"].to_list()
            if N is None: local_N = len(y)
            else:
                local_N = min(len(y), N)
                y = sorted(y, reverse=True)[:local_N]
            x = [method] * local_N
            fig.add_trace(generic_scatter(x = x, y = y, idx = idx,
                showlegend = showlegend))
            showlegend = False

    return fig

def f1_macro_per_n_sample_and_model(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "n_samples", "f1_macro"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model","n_samples"],
        column = "f1_macro",
        N = N
    )
    listOfModels = SUL_string(to_print["model"])
    listOfnSamples = SUL_string(to_print["n_samples"])
    nModels = len(listOfModels)

    multiple_figures_layout(fig, nModels,listOfModels, 
        xaxis_kwargs = {'categoryorder' : "trace", 'type' : "category"})

    # Create bars for each model and method
    to_print["upper_band"] = (to_print["CI_f1_macro_upper"] - 
                              to_print["f1_macro_mean"])
    to_print["lower_band"] = -(to_print["CI_f1_macro_lower"] - 
                              to_print["f1_macro_mean"])
    
    grouped = to_print.groupby(["model","n_samples"])
    for idx, model in enumerate(listOfModels): 
        for n_samples in listOfnSamples : 
            sub_df = grouped.get_group((model,n_samples))
            fig.add_trace(generic_bar(
                sub_df = sub_df, 
                col_x = "n_samples", 
                col_y = "f1_macro_mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = n_samples, 
                idx = idx
            ))

    # Create markers for each model and method
    grouped = selected_data.groupby(["model","n_samples"])
    showlegend = True
    for idx, model in enumerate(listOfModels):  
        for n_samples in listOfnSamples:
            sub_df = grouped.get_group((model,n_samples))
            y = sub_df["f1_macro"].to_list()
            if N is None: local_N = len(y)
            else:
                local_N = min(len(y), N)
                y = sorted(y, reverse=True)[:local_N]
            x = [n_samples] * local_N
            fig.add_trace(generic_scatter(x = x, y = y, idx = idx, 
                showlegend = showlegend))
            showlegend = False
    
    return fig

def f1_macro_per_n_sample_and_method(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["method", "n_samples", "f1_macro"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["method","n_samples"],
        column = "f1_macro",
        N = N
    )
    listOfMethods = SUL_string(to_print["method"])
    # Remove "MLPClassifier (HF)" on purpose
    listOfMethods = [method for method in listOfMethods if method != "MLPClassifier (HF)"]
    listOfnSamples = SUL_string(to_print["n_samples"])
    # Remove "Entraînement Hugging Face" on purpose
    listOfnSamples = [n_samples for n_samples in listOfnSamples if n_samples != "Entraînement Hugging Face"]
    nMethods = len(listOfMethods)

    multiple_figures_layout(fig, nMethods,listOfMethods, 
        xaxis_kwargs = {'categoryorder' : "trace", 'type' : "category"})

    # Create bars for each model and method
    to_print["upper_band"] = (to_print["CI_f1_macro_upper"] - 
                              to_print["f1_macro_mean"])
    to_print["lower_band"] = -(to_print["CI_f1_macro_lower"] - 
                              to_print["f1_macro_mean"])
    
    grouped = to_print.groupby(["method","n_samples"])
    for idx, method in enumerate(listOfMethods): 
        for n_samples in listOfnSamples : 
            sub_df = grouped.get_group((method,n_samples))
            fig.add_trace(generic_bar(
                sub_df = sub_df, 
                col_x = "n_samples", 
                col_y = "f1_macro_mean",
                col_band_u = "upper_band", 
                col_band_l = "lower_band",
                name = n_samples, 
                idx = idx
            ))

    # Create markers for each model and method
    grouped = selected_data.groupby(["method","n_samples"])
    showlegend = True
    for idx, method in enumerate(listOfMethods):  
        for n_samples in listOfnSamples:
            sub_df = grouped.get_group((method,n_samples))
            y = sub_df["f1_macro"].to_list()
            if N is None: local_N = len(y)
            else:
                local_N = min(len(y), N)
                y = sorted(y, reverse=True)[:local_N]
            x = [n_samples] * local_N
            fig.add_trace(generic_scatter(x = x, y = y, idx = idx, 
                showlegend = showlegend))
            showlegend = False
    
    return fig

def f1_macro_lr_per_model_and_method(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "method", "f1_macro", "lr", "epoch", "n_samples", "iteration"]]
    # keep, per model, method and lr, the best result for each epoch
    selected_data = selected_data.groupby(["model","method","lr", "n_samples", "iteration"]).\
        agg(f1_macro = ("f1_macro", lambda col : max(col)))
    
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model","method", "lr"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    nModels = len(listOfModels)

    multiple_figures_layout(fig, nModels,listOfModels, 
        xaxis_kwargs={'tickvals' : [5e-6, 1e-5,2e-5,5e-5], 'range' : [-5.4,-4.2],\
                      'type' : "log"},
        xlabel_prefix = "Learning rate<br><br>")

    # Create bars for each model and method
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))
            traces = generic_scatter_with_bands(
                df = sub_df, 
                col_x = "lr", 
                col_y = "f1_macro_mean", 
                col_u = "CI_f1_macro_upper", 
                col_l = "CI_f1_macro_lower", 
                name = method, 
                idx = idx
            )

            fig.add_trace(traces[0])
            fig.add_trace(traces[1])
    
    return fig

def f1_macro_lr_per_model(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    # Create figure 
    fig = Figure(layout = layout_general_parameters)
    one_figure_layout(fig, "Learning Rate", xaxis_kwargs={"type" : "log"})

    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "f1_macro", "lr"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model", "lr"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])


    # Create bars for each model 
    grouped = to_print.groupby(["model"])
    for model in listOfModels: 
        sub_df = grouped.get_group((model,))
        traces = generic_scatter_with_bands(
            df = sub_df, 
            col_x = "lr", 
            col_y = "f1_macro_mean", 
            col_u = "CI_f1_macro_upper", 
            col_l = "CI_f1_macro_lower", 
            name = model, 
        )

        fig.add_trace(traces[0])
        fig.add_trace(traces[1])
    return fig

def f1_macro_epoch_per_model_and_method(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "method", "f1_macro", "epoch"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model","method", "epoch"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    nModels = len(listOfModels)

    multiple_figures_layout(fig, nModels,listOfModels, 
        xaxis_kwargs={})

    # Create bars for each model and method
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))
            traces = generic_scatter_with_bands(
                df = sub_df, 
                col_x = "epoch", 
                col_y = "f1_macro_mean", 
                col_u = "CI_f1_macro_upper", 
                col_l = "CI_f1_macro_lower", 
                name = method, 
                idx = idx
            )

            fig.add_trace(traces[0])
            fig.add_trace(traces[1])
    
    return fig

def f1_macro_epoch_per_model(
        df : pd.DataFrame, 
        N : int|None = None) -> Figure:
    # Create figure 
    fig = Figure(layout = layout_general_parameters)
    one_figure_layout(fig, "Nombre d'epochs")

    # Process Data
    selected_data : pd.DataFrame = df.loc[:,["model", "f1_macro", "epoch"]]
    to_print : pd.DataFrame = get_mean_and_half_band(
        df = selected_data,
        groupbyColumns = ["model", "epoch"],
        column = "f1_macro",
        N = N
    )
    listOfModels = SUL_string(to_print["model"])

    # Create bars for each model 
    grouped = to_print.groupby(["model"])
    for model in listOfModels: 
        sub_df = grouped.get_group((model,))
        traces = generic_scatter_with_bands(
            df = sub_df, 
            col_x = "epoch", 
            col_y = "f1_macro_mean", 
            col_u = "CI_f1_macro_upper", 
            col_l = "CI_f1_macro_lower", 
            name = model
        )

        fig.add_trace(traces[0])
        fig.add_trace(traces[1])
    return fig

def SUL_string(vec) : 
    '''return a sorted list of unique string items'''
    return sorted(list(set(vec)), key = lambda x : str(x).lower())

def multiple_figures_layout(
        fig : Figure, 
        nFrames : int, 
        listOfFramesNames : list[str],  
        xaxis_kwargs : dict = {},   
        xlabel_prefix : str = ""
    ):
    if nFrames >1:
        subplot_width =  0.9 / nFrames
        gap = 0.1 / (nFrames - 1)
        fig.update_layout({
            'xaxis'  : {
                'anchor' : "x1" , 
                'domain' : [0.0,subplot_width],
                'title' : {'text' : f"{xlabel_prefix} {listOfFramesNames[0]}"},
                'gridcolor' : gridcolor_x,
                "zerolinecolor" : gridcolor_x,
                **xaxis_kwargs
            },
            "yaxis_title_text" : "Score F1 macro",
            "yaxis_range" : [0,1],
            **{
                f'xaxis{i+1}' : {
                    'anchor' : f"x{i+1}" , 
                    'domain' : [
                        (subplot_width + gap) * i, 
                        min((subplot_width + gap) * i + subplot_width, 1)
                        ],
                    'title' : {'text' : f"{xlabel_prefix} {listOfFramesNames[i]}"},
                    'gridcolor' : gridcolor_x,
                    "zerolinecolor" : gridcolor_x,
                    **xaxis_kwargs
                }
                for i in range(1, nFrames)
            }
        })
    else : 
        fig.update_layout({
            'xaxis'  : {
                'anchor' : "x1" , 
                'domain' : [0.0,1],
                'title' : {'text' : listOfFramesNames[0]},
                'gridcolor' : gridcolor_x,
                "zerolinecolor" : gridcolor_x,
                **xaxis_kwargs
            },
            "yaxis_title_text" : "Score F1 macro",
            "yaxis_range" : [0,1]
        })

def one_figure_layout(
        fig : Figure, 
        xLabel : str,
        xaxis_kwargs :dict = {}   
    ):
    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,1.0],
            'title' : {'text' : xLabel},
            'gridcolor' : gridcolor_x,
            "zerolinecolor" : gridcolor_x,
            **xaxis_kwargs
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
    })

def generic_bar(
        sub_df : pd.DataFrame, 
        col_x : str, 
        col_y : str, 
        col_band_u : str, 
        col_band_l : str,
        name : str, 
        idx : int
    ):
    return go.Bar(
        x = sub_df[col_x],
        y = sub_df[col_y],
        error_y = {
            'type' : "data", 
            'symmetric' : False,
            'array' : sub_df[col_band_u],
            'arrayminus' : sub_df[col_band_l],
            'color' : colors["error"],
            'width' : 20,
            'thickness' : 2
        },
        name = name,
        marker = {
            'color' : colors[name],
            'cornerradius' : 5
        },
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = (idx == 0)
    )

def generic_scatter(
        x : str,
        y : str,
        idx : str,
        name : str = "Modèle d'inférence unique",
        showlegend : bool = False
    ):

    return go.Scatter(
        x = x,
        y = y,
        marker = {
            'color' : colors["marker"],
            'opacity' : 0.4,
            'symbol' : "circle"
        },
        mode = "markers",
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        name = name,
        showlegend = showlegend
    )

def error_band_color(rgb_color, error_band_opacity = 0.2):
    out = rgb_color[:3] + "a" + rgb_color[3:-1] + f",{error_band_opacity})"
    return out

def add_baselines(fig : Figure, width : int, n_subplots : int) : 
    for i in range(n_subplots) : 
        for key, value in meilleurs_models.items() : 
            fig.add_shape(type="line",
                x0 = -1,
                x1 = width, 
                xref = f"x{i+1}",
                y0 = value, 
                y1 = value, 
                yref = "y",
                line={'color':"red",'width':2, 'dash' :dash_meilleurs_models[key]}
            )

def generic_scatter_with_bands(df, col_x, col_y, col_u, col_l, name, idx = 0):
    trace_1 = go.Scatter(
        x = df[col_x],
        y = df[col_y],
        name = name,
        marker = {
            'color' : colors[name],
        },
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = (idx == 0),
        zorder = 1
    )

    trace_2 = go.Scatter(
        x = [*df[col_x],*df[col_x][::-1]],
        y = [*df[col_u], *df[col_l][::-1]],
        name = name,
        xaxis = f"x{idx + 1}",
        yaxis = "y",
        showlegend = False,
        #Filling
        fill ='toself',
        fillcolor = error_band_color(colors[name]),
        line = dict(color='rgba(0,0,0,0)'),
        hoverinfo = "skip",
        zorder = 0
    )
    return trace_1, trace_2