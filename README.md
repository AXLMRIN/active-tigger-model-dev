# active-tigger-model-dev
This repository is used to develop models to be used in [active tigger](https://github.com/emilienschultz/activetigger)

# Env
```
conda create -n VENV python=3.11
pip install -q transformers datasets
conda install pytorch
pip install 'accelerate>=0.26.0'
conda install s3fs
conda install scikit-learn
```
Si sur GPU : 
```
pip install -U flash_attn
```

CAREFUL, this is not running properly on some GPUs — only god knows why.