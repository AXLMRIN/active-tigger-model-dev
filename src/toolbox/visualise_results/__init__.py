# Constants ######################################################################
GRIDCOLOR_X = "rgba(100,100,100,0.5)"
LAYOUT = {
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

COLORS = {
    "KNeighborsClassifier"    : "rgb(230,  57,  71)", 
    "RandomForestClassifier"  : "rgb(255, 183,   3)", 
    "MLPClassifier"           : "rgb( 69, 123, 157)", 
    "Baseline - HF Classifier": "rgb(144, 190, 109)", 
    
    'answerdotai/ModernBERT-base'   : "rgb(255,   0,  85)", 
    'FacebookAI/roberta-base'       : "rgb(255,  85,   0)", 
    'google-bert/bert-base-uncased' : "rgb( 56,   0, 153)",

    100                         : "rgb(144, 225, 239)",
    200                         : "rgb( 72, 202, 228)",
    300                         : "rgb(  0, 180, 216)",
    400                         : "rgb(  0, 149, 199)",
    500                         : "rgb(  0, 118, 182)",
    1000                        : "rgb(  2,  61, 138)",
    1500                        : "rgb(  3,   5,  94)",
    2000                        : "rgb(  0,   0,   0)",
    "Entraînement Hugging Face" : "rgb(255, 209, 102)", 

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
# IMPORTS ######################################################################
from .VisualiseAll import VisualiseAll