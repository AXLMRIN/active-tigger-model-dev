# active-tigger-model-dev
This repository is used to develop models to be used in [active tigger](https://github.com/emilienschultz/activetigger)

# Env
```
conda create -n VENV python=3.11.11
pip install -q transformers datasets
conda install pytorch
pip install 'accelerate>=0.26.0'
conda install s3fs
```
Si sur GPU : 
```
pip install -U flash_attn
```

# Data
```python
import pandas as pd

df = pd.read_parquet("hf://datasets/nateraw/rap-lyrics-v2/data/train-00000-of-00001-0d87a5bc980f4999.parquet")
```