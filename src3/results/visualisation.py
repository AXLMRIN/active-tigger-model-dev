from jinja2 import Template
import pandas as pd

import plotly.graph_objects as go

import visualiseThis as vt




data = vt.genData({
    "RandomForest" : "2025-05-18-RandomForest-2.csv",
    "KNN" : "2025-05-18-KNN-2.csv",
    "OneLayer" : "2025-05-18-MLPOneLayer-2.csv"
})
print(data.columns())
all_figures = {}


vt.export(all_figures)