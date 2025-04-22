# active-tigger-model-dev
This repository is used to develop models to be used in [active tigger](https://github.com/emilienschultz/activetigger)

# Env
```bash
conda create -n VENV python=3.11
pip install -q transformers datasets
conda install pytorch
pip install 'accelerate>=0.26.0'
conda install s3fs
conda install scikit-learn
pip install mergedeep
conda install plotly
pip install -U kaleido
pip install pygad
```
Si sur GPU : 
```
pip install -U flash_attn
```

CAREFUL, this is not running properly on some GPUs — only god knows why.

# Optimization with SMAC3
seems to be linux only
```bash
conda create -n AT-HPO python=3.10
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
```
