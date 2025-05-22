from jinja2 import Template
import pandas as pd

import plotly.graph_objects as go

import visualiseThis as vt




data = vt.fetch_data(
    {
    "Random Forest" : "2025-05-18-RandomForest-2.csv",
    "KNN" : "2025-05-18-KNN-2.csv",
    "MLPClassifier (slkearn)" : "2025-05-18-MLPOneLayer-2.csv",
    "MLPClassifier (HF)" : "2025-05-20-HuggingFaceClassification.csv"
    },
    concat_col = "method",
    usecols = ["filename", "n_samples", "epoch", "f1_macro"]
)

N_best = 200

all_figures = {}

all_figures["comparaison_plongement_classification_all"] = \
    vt.f1_macro_per_model_and_method(data, N = None)
all_figures["comparaison_plongement_classification_N_best"] = \
    vt.f1_macro_per_model_and_method(data, N = N_best)

all_figures["comparaison_plongement_n_sample_all"] = \
    vt.f1_macro_per_n_sample_and_method(data, N = None)
all_figures["comparaison_plongement_n_sample_N_best"] = \
    vt.f1_macro_per_n_sample_and_method(data, N = N_best)

all_figures["comparaison_lr_model_method_all"] = \
    vt.f1_macro_lr_per_model_and_method(data, N = None)
all_figures["comparaison_lr_model_method_N_best"] = \
    vt.f1_macro_lr_per_model_and_method(data, N = N_best)
all_figures["comparaison_lr_model_all"] = \
    vt.f1_macro_lr_per_model(data, N = None)
all_figures["comparaison_lr_model_N_best"] = \
    vt.f1_macro_lr_per_model(data, N = N_best)

all_figures["comparaison_epoch_model_method_all"] = \
    vt.f1_macro_epoch_per_model_and_method(data, N = None)
all_figures["comparaison_epoch_model_method_N_best"] = \
    vt.f1_macro_epoch_per_model_and_method(data, N = N_best)
all_figures["comparaison_epoch_model_all"] = \
    vt.f1_macro_epoch_per_model(data, N = None)
all_figures["comparaison_epoch_model_N_best"] = \
    vt.f1_macro_epoch_per_model(data, N = N_best)

vt.export(all_figures, N_best)