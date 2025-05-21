from jinja2 import Template
import pandas as pd

import plotly.graph_objects as go

import visualiseThis as vt




data = vt.genData({
    "RandomForest" : "2025-05-18-RandomForest-2.csv",
    "KNN" : "2025-05-18-KNN-2.csv",
    "OneLayer" : "2025-05-18-MLPOneLayer-2.csv"
})

all_figures = {}
all_figures["comparaison_plongement_classification_all"] = \
    vt.f1_macro_per_model_and_method(data, N = None)
all_figures["comparaison_plongement_classification_200"] = \
    vt.f1_macro_per_model_and_method(data, N = 200)
all_figures["comparaison_lr_model_method_all"] = \
    vt.f1_macro_lr_per_model_and_method(data, N = None)
all_figures["comparaison_lr_model_method_200"] = \
    vt.f1_macro_lr_per_model_and_method(data, N = 200)
# all_figures["comparaison_lr_all"] = \
#     vt.f1_macro_lr_per_model_and_method(data, N = None)
# all_figures["comparaison_lr_200"] = \
#     vt.f1_macro_lr_per_model_and_method(data, N = 200)

vt.export(all_figures)