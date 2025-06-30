
# Text Classification: exploring hyper parameters and classifying techniques

## TLDR

- **🎯 Goal:** Exploring the impact of hyperparameter tuning and classifying techniques on text classification performances. 
- **🗂️ Framework:** The classification tasks were chosen to fit in the CSS[^1] context. The tasks were gatheres by [Ziems et al. (2024)](https://direct.mit.edu/coli/article/50/1/237/118498/Can-Large-Language-Models-Transform-Computational).
- **⚙️ How:** Proposing a pipeline allowing users to choose a set of parameters to test and visualise the results for further analysis. The pipeline uses state of the art libraries such as 🤗 Hugging Face transformers, PyTorch and Scikit-learn.
- **💻 Setup:** The repository contains an `environment.yml` file for easy setup. The pipeline can be run on CPU or (single) GPU.

## Setting up the environment 

To set up the environment, use the `requirements.yml` file as such : 

```bash
conda env create -f requirements_linux.yml -n ENV_NAME
```

or

```bash
conda env create -f requirements_mac.yml -n ENV_NAME
```

Alternatively, you can create the environment manually by typing : 

```bash
conda create -n ENV_NAME python=3.11
conda activate ENV_NAME
pip install -qU transformers datasets accelerate mergedeep pygad kaleido great-tables selenium
conda install pytorch scikit-learn plotly pandas
```


## Libraries used

- [Pandas](https://pandas.pydata.org/docs/), [Numpy](https://numpy.org/doc/2.3/) and [Scipy](https://docs.scipy.org/doc/scipy/): Used for common data management and statistic calculus.
- [🤗 Hugging Face - transformers and datasets](https://huggingface.co/docs): This library is used to load data and train the embedding model.
- [Scikit-learn](https://scikit-learn.org/stable/): Used to load, train and test classifiers used on embeddings.
- [Pygad](https://pygad.readthedocs.io/en/latest/): Used to optimise the hyperparameters of the Scikit-learn classifiers.

## The pipeline

### Step 1: Train an embedding model

With the class `DataHandler`, we open a csv file, preprocessa and split data (possible stratification) into a train, eval and test set.

> **Parameters to set:** data, preprocessing, stratification.

With the class `CustomTransformersPipeline` load the model and tokenizer, encode the test, eval and train set and then train the embedding model with the classic hugging face API.

> **Parameters to set:** embedding model, context window size, number of epochs, optimizer, learning rate, weight decay, warmup ratio.

**What is saved during this step:** 

- Full checkpoint during the training (for each epoch). Files saved in `./models/model/name/iteration_XXX/checkpoint-XXX`.
- the model name in `./models/model/iteration_XXX/name`.
- the encoded train, eval and test set are saved in `./models/model/name/iteration_XXX/data`

### Step 2: Test the model after each epoch

With the `TestAllEpochs` object, we load the model after each epoch and the encoded data previously saved and test the model on the test set.

> **Parameters to set:** 

**What is saved during this step:**

- A csv file of the results in `./results`

### Step 3: Export the embeddings

With the `ExportEmbeddingsForAllEpochs` object, we load the model after each epoch and the encoded data previously saved, we concatenate the train and eval set and embed all entries. It is possible to delete the heavy files from each checkpoint after the embeddings are exported.

> **Parameters to set:** 

**What is saved during this step:**

- The embeddings (for the train and test set) in `./models/model/name/iteration_XXX/embeddings`.
- The labels (for the train and test set) in `./models/model/name/iteration_XXX/embeddings`.

### Step 4: Optimise scikit-learn classifiers

With the `DataHandlerForGOfSC` object, we load the embeddings and labels (for the train and test set) previously saved.

> **Parameters to set:**

With the `RoutineGOfSC` object _(Routine Genetic Optimisation for Scikit-learn Classifiers)_, we maximise f1-macro on the test set by optimising the hyperparameters of scikit-learn classifiers with a genetic algorithm. Two presets are available (`RoutineGOfKNN`, `RoutineGOfRF`) for optimising a KNN and a Random Forest classifier.

> **Parameters to set:** hyperparameters (gene) space, other parameters related to the genetic algorithm.

**What is saved during this step:**

- a csv file of the score and the optimised hyperparameters in `./results/`.

### Step 5: Visualise your results

With the `Table` object, we can create tables by giving a csv file with the baseline results (ie the results of Step 2), a csv file with other results (ie the results of Step 4) and at least 2 columns ("row" and "column" columns). The program will evaluate the mean and the confidence intervals for each cell.  

> **Parameters to set:** 

With the `Table` object, we can create tables by giving a csv file with the baseline results (ie the results of Step 2), a csv file with other results (ie the results of Step 4) and up to 3 columns ("trace" for different colors, "frame" to split subplots and "x_axis" to specify the x axis). If "x_axis" is given, the plot will be a line plot, however, if it isn't it will be a barplot. The program will evaluate the mean and the confidence intervals for each point.  

> **Parameters to set:** 

**What is saved during this step:**

- as many visualisations as you like, all saved in `./figures` as an html or png.

## The data architecture


- **./**
    - **📔 data**
        - Your choice
    
    - **💻 src**
        - toolbox
            - All classes and functions
        - Example of routines

    - **🚀 models**
        - model name prefix (ex: FacebookAI)
            - model name suffix (ex: roberta-base)
                - iteration (ex: 001)
                - ...

    - **📋 results**
        - ... Your choice
    
    - **📖 pers_logs**
        - ... Your choice
    
    - **📊 figures**
        - ... Your choice

[^1]: Computational Social Science.