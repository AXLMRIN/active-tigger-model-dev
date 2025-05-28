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
onlyDisplayBestEpoch = True 
all_figures = {}

# vt.estimate_time_to_process(data)

all_figures["comparaison_plongement_classification_all"] = \
    vt.f1_macro_per_model_and_method(data, N = None, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

all_figures["f1_HF_vs_f1_autre"] = \
    vt.f1_macro_f1_hf_per_model_and_method(data, N = None)

vt.f1_macro_per_n_sample_and_method_Table(
    data, N = None, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

all_figures["comparaison_lr_model_method_all"] = \
    vt.f1_macro_lr_per_model_and_method(data, N = None, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

vt.f1_macro_epoch_per_model_and_method_Table(data, N = None)

vt.export(all_figures)