# === === === === === === === === === === === === === === === === === === === ==
# IMPORTS
# === === === === === === === === === === === === === === === === === === === ==
from .data_handling import genData
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
import pandas as pd

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
    "KNN"                    : "rgb(230, 57, 71)", 
    "Random Forest"          : "rgb(255, 183, 3)", 
    "MLPClassifier (slkearn)" : "rgb(69, 123, 157)", 
    "MLPClassifier (HF)"      : "rgb(130, 130, 130)", 
    
    'answerdotai/ModernBERT-base'   : "rgb(255, 0, 85)", 
    'FacebookAI/roberta-base'       : "rgb(255, 85, 0)", 
    'google-bert/bert-base-uncased' : "rgb(56, 0, 153)",

    500   : "rgb(144, 225, 239)",
    1000  : "rgb(0, 180, 216)",
    1500  : "rgb(2, 61, 138)",

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

def f1_macro_per_model_and_method(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "filename_csv", "f1_macro"])
    selected_data.rename(columns = {"filename_csv" : "method"})
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model","method"],
        column = "f1_macro",
        N = N
    )
    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    N_models = len(listOfModels)

    # Create figure 
    subplot_width =  0.9 / N_models
    gap = 0.1 / (N_models - 1)

    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,subplot_width],
            'title' : {
                'text' : listOfModels[0]
            },
            'categoryorder' : "trace"
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
        **{
            f'xaxis{i+1}' : {
                'anchor' : f"x{i+1}" , 
                'domain' : [
                    (subplot_width + gap) * i, 
                    (subplot_width + gap) * i + subplot_width
                    ],
                'title' : {'text' : listOfModels[i]},
                'categoryorder' : "trace"
            }
            for i in range(1, N_models)
        }
    })

    # Create bars for each model and method
    to_print["upper_band"] = (to_print["CI_f1_macro_upper"] - 
                              to_print["f1_macro_mean"])
    to_print["lower_band"] = -(to_print["CI_f1_macro_lower"] - 
                              to_print["f1_macro_mean"])
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))
            fig.add_trace(
                go.Bar(
                    x = sub_df["method"],
                    y = sub_df["f1_macro_mean"],
                    error_y = {
                        'type' : "data", 
                        'symmetric' : False,
                        'array' : sub_df["upper_band"],
                        'arrayminus' : sub_df["lower_band"],
                        'color' : colors["error"],
                        'width' : 20,
                        'thickness' : 2
                    },
                    name = method,
                    marker = {
                        'color' : colors[method],
                        'cornerradius' : 5
                    },
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = (idx == 0)
                )
            )

    # Create markers for each model and method
    grouped = df.to_DataFrame().groupby(["model"])
    for idx, model in enumerate(listOfModels):  
        sub_df = grouped.get_group((model,))
        fig.add_trace(
            go.Scatter(
                x = sub_df["filename_csv"],
                y = sub_df["f1_macro"],
                marker = {
                    'color' : colors["marker"],
                    'opacity' : 0.4,
                    'symbol' : "circle"
                },
                mode = "markers",
                xaxis = f"x{idx + 1}",
                yaxis = "y",
                name = "modèle d'inférence unique",
                showlegend = (idx == 0)
            )
        )

    return fig

def f1_macro_per_n_sample_and_method(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "n_samples", "f1_macro"])
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model","n_samples"],
        column = "f1_macro",
        N = N
    )
    listOfModels = SUL_string(to_print["model"])
    listOfnSamples = list(set(to_print["n_samples"]))
    N_models = len(listOfModels)

    # Create figure 
    subplot_width =  0.9 / N_models
    gap = 0.1 / (N_models - 1)

    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,subplot_width],
            'title' : {
                'text' : listOfModels[0]
            },
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
        **{
            f'xaxis{i+1}' : {
                'anchor' : f"x{i+1}" , 
                'domain' : [
                    (subplot_width + gap) * i, 
                    (subplot_width + gap) * i + subplot_width
                    ],
                'title' : {'text' : listOfModels[i]},
            }
            for i in range(1, N_models)
        }
    })

    # Create bars for each model and method
    to_print["upper_band"] = (to_print["CI_f1_macro_upper"] - 
                              to_print["f1_macro_mean"])
    to_print["lower_band"] = -(to_print["CI_f1_macro_lower"] - 
                              to_print["f1_macro_mean"])
    grouped = to_print.groupby(["model","n_samples"])
    for idx, model in enumerate(listOfModels): 
        for n_samples in listOfnSamples : 
            sub_df = grouped.get_group((model,n_samples))
            fig.add_trace(
                go.Bar(
                    x = sub_df["n_samples"],
                    y = sub_df["f1_macro_mean"],
                    error_y = {
                        'type' : "data", 
                        'symmetric' : False,
                        'array' : sub_df["upper_band"],
                        'arrayminus' : sub_df["lower_band"],
                        'color' : colors["error"],
                        'width' : 20,
                        'thickness' : 2
                    },
                    name = n_samples,
                    marker = {
                        'color' : colors[n_samples],
                        'cornerradius' : 5
                    },
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = (idx == 0)
                )
            )

    # Create markers for each model and method
    grouped = df.to_DataFrame().groupby(["model"])
    for idx, model in enumerate(listOfModels):  
        sub_df = grouped.get_group((model,))
        fig.add_trace(
            go.Scatter(
                x = sub_df["n_samples"],
                y = sub_df["f1_macro"],
                marker = {
                    'color' : colors["marker"],
                    'opacity' : 0.4,
                    'symbol' : "circle"
                },
                mode = "markers",
                xaxis = f"x{idx + 1}",
                yaxis = "y",
                name = "modèle d'inférence unique",
                showlegend = (idx == 0)
            )
        )
    
    return fig

def f1_macro_lr_per_model_and_method(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "filename_csv", "f1_macro", "lr"])
    selected_data.rename(columns = {"filename_csv" : "method"})
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model","method", "lr"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    N_models = len(listOfModels)

    # Create figure 
    subplot_width =  0.9 / N_models
    gap = 0.1 / (N_models - 1)

    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,subplot_width],
            'title' : {
                'text' : listOfModels[0]
            },
            'type' : 'log',
            'gridcolor' : gridcolor_x,
            "zerolinecolor" : gridcolor_x,
            'tickvals' : [5e-6, 1e-5,2e-5,5e-5], 
            'range' : [-5.4,-4.2]
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
        **{
            f'xaxis{i+1}' : {
                'anchor' : f"x{i+1}" , 
                'domain' : [
                    (subplot_width + gap) * i, 
                    (subplot_width + gap) * i + subplot_width
                    ],
                'title' : {'text' : listOfModels[i]},
                'type' : 'log',
                'gridcolor' : gridcolor_x,
                "zerolinecolor" : gridcolor_x,
                'tickvals' : [5e-6, 1e-5,2e-5,5e-5], 
                'range' : [-5.4,-4.2]
            }
            for i in range(1, N_models)
        }
    })

    # Create bars for each model and method
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))

            

            fig.add_trace(
                go.Scatter(
                    x = sub_df["lr"],
                    y = sub_df["f1_macro_mean"],
                    name = method,
                    marker = {
                        'color' : colors[method],
                    },
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = (idx == 0),
                    zorder = 1
                )
            )

            fig.add_trace(
                go.Scatter(
                    x = [*sub_df["lr"],*sub_df["lr"][::-1]],
                    y = [*sub_df["CI_f1_macro_upper"], *sub_df["CI_f1_macro_lower"][::-1]],
                    name = method,
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = False,
                    #Filling
                    fill ='toself',
                    fillcolor = error_band_color(colors[method]),
                    line = dict(color='rgba(0,0,0,0)'),
                    hoverinfo = "skip",
                    zorder = 0
                )
            )
    
    return fig

def f1_macro_lr_per_model(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "f1_macro", "lr"])
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model", "lr"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])
    N_models = len(listOfModels)

    # Create figure 
    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,1.0],
            'title' : {'text' : "Learning rate"},
            'type' : 'log',
            'gridcolor' : gridcolor_x,
            "zerolinecolor" : gridcolor_x,
            'tickvals' : [5e-6, 1e-5,2e-5,5e-5], 
            'range' : [-5.4,-4.2]
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
    })

    # Create bars for each model 
    grouped = to_print.groupby(["model"])
    for idx, model in enumerate(listOfModels): 
        sub_df = grouped.get_group((model,))

        fig.add_trace(
            go.Scatter(
                x = sub_df["lr"],
                y = sub_df["f1_macro_mean"],
                name = model,
                marker = {
                    'color' : colors[model],
                },
                xaxis = "x1",
                yaxis = "y",
                showlegend = True,
                zorder = 1
            )
        )

        fig.add_trace(
            go.Scatter(
                x = [*sub_df["lr"],*sub_df["lr"][::-1]],
                y = [*sub_df["CI_f1_macro_upper"], *sub_df["CI_f1_macro_lower"][::-1]],
                name = model,
                xaxis = "x1",
                yaxis = "y",
                showlegend = False,
                #Filling
                fill ='toself',
                fillcolor = error_band_color(colors[model]),
                line = dict(color='rgba(0,0,0,0)'),
                hoverinfo = "skip",
                zorder = 0
            )
        )

    return fig

def f1_macro_epoch_per_model_and_method(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "filename_csv", "f1_macro", "epoch"])
    selected_data.rename(columns = {"filename_csv" : "method"})
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model","method", "epoch"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])
    listOfMethods = SUL_string(to_print["method"])
    N_models = len(listOfModels)

    # Create figure 
    subplot_width =  0.9 / N_models
    gap = 0.1 / (N_models - 1)

    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,subplot_width],
            'title' : {
                'text' : listOfModels[0]
            },
            'gridcolor' : gridcolor_x,
            "zerolinecolor" : gridcolor_x
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
        **{
            f'xaxis{i+1}' : {
                'anchor' : f"x{i+1}" , 
                'domain' : [
                    (subplot_width + gap) * i, 
                    (subplot_width + gap) * i + subplot_width
                    ],
                'title' : {'text' : listOfModels[i]},
                'gridcolor' : gridcolor_x,
                "zerolinecolor" : gridcolor_x
            }
            for i in range(1, N_models)
        }
    })

    # Create bars for each model and method
    grouped = to_print.groupby(["model","method"])
    for idx, model in enumerate(listOfModels): 
        for method in listOfMethods : 
            sub_df = grouped.get_group((model,method))

            

            fig.add_trace(
                go.Scatter(
                    x = sub_df["epoch"],
                    y = sub_df["f1_macro_mean"],
                    name = method,
                    marker = {
                        'color' : colors[method],
                    },
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = (idx == 0),
                    zorder = 1
                )
            )

            fig.add_trace(
                go.Scatter(
                    x = [*sub_df["epoch"],*sub_df["epoch"][::-1]],
                    y = [*sub_df["CI_f1_macro_upper"], *sub_df["CI_f1_macro_lower"][::-1]],
                    name = method,
                    xaxis = f"x{idx + 1}",
                    yaxis = "y",
                    showlegend = False,
                    #Filling
                    fill ='toself',
                    fillcolor = error_band_color(colors[method]),
                    line = dict(color='rgba(0,0,0,0)'),
                    hoverinfo = "skip",
                    zorder = 0
                )
            )
    
    return fig

def f1_macro_epoch_per_model(
        df : genData, 
        N : int|None = None) -> Figure:
    
    fig = Figure(layout = layout_general_parameters)

    # Process Data
    selected_data : genData = df.pick(["model", "f1_macro", "epoch"])
    to_print : pd.DataFrame = selected_data.get_mean_and_half_band(
        groupbyColumns = ["model", "epoch"],
        column = "f1_macro",
        N = N
    )

    listOfModels = SUL_string(to_print["model"])

    # Create figure 
    fig.update_layout({
        'xaxis'  : {
            'anchor' : "x1" , 
            'domain' : [0.0,1.0],
            'title' : {'text' : "Nombre d'epochs"},
            'gridcolor' : gridcolor_x,
            "zerolinecolor" : gridcolor_x
        },
        "yaxis_title_text" : "Score F1 macro",
        "yaxis_range" : [0,1],
    })

    # Create bars for each model 
    grouped = to_print.groupby(["model"])
    for idx, model in enumerate(listOfModels): 
        sub_df = grouped.get_group((model,))

        fig.add_trace(
            go.Scatter(
                x = sub_df["epoch"],
                y = sub_df["f1_macro_mean"],
                name = model,
                marker = {
                    'color' : colors[model],
                },
                xaxis = "x1",
                yaxis = "y",
                showlegend = True,
                zorder = 1
            )
        )

        fig.add_trace(
            go.Scatter(
                x = [*sub_df["epoch"],*sub_df["epoch"][::-1]],
                y = [*sub_df["CI_f1_macro_upper"], *sub_df["CI_f1_macro_lower"][::-1]],
                name = model,
                xaxis = "x1",
                yaxis = "y",
                showlegend = False,
                #Filling
                fill ='toself',
                fillcolor = error_band_color(colors[model]),
                line = dict(color='rgba(0,0,0,0)'),
                hoverinfo = "skip",
                zorder = 0
            )
        )

    return fig


def SUL_string(vec) : 
    '''return a sorted list of unique string items'''
    return sorted(list(set(vec)), key = lambda x : x.lower())