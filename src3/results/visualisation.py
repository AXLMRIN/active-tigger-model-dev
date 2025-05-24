from jinja2 import Template
import pandas as pd

import plotly.graph_objects as go

import visualiseThis as vt



#FIXME les visus sont pétés j'en peux plus bon courage à toi du futur

data = vt.fetch_data(
    {
        "Random Forest" : "2025-05-24-RandomForest-F.csv",
        "KNN" : "2025-05-24-KNN-F.csv",
        "MLPClassifier (slkearn)" : "2025-05-24-MLPOneLayer-F.csv",
        "MLPClassifier (HF)" : "2025-05-24-HuggingFaceClassification-F.csv"
    },
    usecols = ["filename", "n_samples", "epoch", "f1_macro", "iteration", "time"]
)

# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===
# TESTS
# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===

# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===
# VISUS
# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===
N_best = 20

all_figures = {}

# vt.estimate_time_to_process(data)

# all_figures["comparaison_plongement_classification_all"] = \
#     vt.f1_macro_per_model_and_method(data, N = None)
# all_figures["comparaison_plongement_classification_N_best"] = \
#     vt.f1_macro_per_model_and_method(data, N = N_best)

all_figures["f1_HF_vs_f1_autre"] = \
    vt.f1_macro__f1_hf_per_model_and_method(data, N = None)
# all_figures["f1_HF_vs_f1_autre"] = \
#     vt.f1_macro__f1_hf_per_model_and_method(data, N = N_best)

# all_figures["comparaison_plongement_n_sample_all"] = \
#     vt.f1_macro_per_n_sample_and_model(data, N = None)
# all_figures["comparaison_plongement_n_sample_N_best"] = \
#     vt.f1_macro_per_n_sample_and_model(data, N = N_best)

# all_figures["comparaison_classification_n_sample_all"] = \
#     vt.f1_macro_per_n_sample_and_method(data, N = None)
# all_figures["comparaison_classification_n_sample_N_best"] = \
#     vt.f1_macro_per_n_sample_and_method(data, N = N_best)

# all_figures["comparaison_lr_model_method_all"] = \
#     vt.f1_macro_lr_per_model_and_method(data, N = None)
# all_figures["comparaison_lr_model_method_N_best"] = \
#     vt.f1_macro_lr_per_model_and_method(data, N = N_best)
# all_figures["comparaison_lr_model_all"] = \
#     vt.f1_macro_lr_per_model(data, N = None)
# all_figures["comparaison_lr_model_N_best"] = \
#     vt.f1_macro_lr_per_model(data, N = N_best)

# all_figures["comparaison_epoch_model_method_all"] = \
#     vt.f1_macro_epoch_per_model_and_method(data, N = None)
# all_figures["comparaison_epoch_model_method_N_best"] = \
#     vt.f1_macro_epoch_per_model_and_method(data, N = N_best)
# all_figures["comparaison_epoch_model_all"] = \
#     vt.f1_macro_epoch_per_model(data, N = None)
# all_figures["comparaison_epoch_model_N_best"] = \
#     vt.f1_macro_epoch_per_model(data, N = N_best)

vt.export(all_figures, N_best)