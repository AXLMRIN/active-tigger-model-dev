from jinja2 import Template
import pandas as pd

import plotly.graph_objects as go

import visualiseThis as vt

data = vt.fetch_data(
    {
        "Random Forest" : "2025-06-09-RandomForest-3.csv",
        "KNN" : "2025-06-09-KNN-3.csv",
        "MLPClassifier (slkearn)" : "2025-06-09-MLPOneLayer-3.csv",
        "MLPClassifier (HF)" : "2025-06-09-HuggingFaceClassification-2.csv"
    },
    usecols = ["filename", "n_samples", "epoch", "f1_macro", "iteration", "time"]
)

# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===
# VISUS
# === === === ===  === ===  === ===  === ===  === ===  === ===  === ===  === ===
onlyDisplayBestEpoch = True 
all_figures = {}

# vt.estimate_time_to_process(data)

all_figures["comparaison_plongement_classification"] = \
    vt.f1_macro_per_model_and_method(data, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

all_figures["f1_HF_vs_f1_autre"] = \
    vt.f1_macro_f1_hf_per_model_and_method(data)

vt.f1_macro_per_n_sample_and_method_Table(data, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

all_figures["comparaison_lr_model_method"] = \
    vt.f1_macro_lr_per_model_and_method(data, onlyDisplayBestEpoch = onlyDisplayBestEpoch)

all_figures["time_n_samples"] = vt.time_n_samples(data)

vt.f1_macro_epoch_per_model_and_method_Table(data)

vt.export(all_figures)