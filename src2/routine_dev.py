# CUSTOM IMPORTS
from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder
from copy import deepcopy
config = Config()
dataset = CustomDataset(config)
dataset.open_dataset()
dataset.find_labels()

def preprocess_function_label(batch, label2id): 
    return [label2id[label] for label in batch["label"]]
def preprocess_function_text(batch): 
    return [sentence.lower() for sentence in batch["text"]]

LABEL2ID = deepcopy(dataset.label2id)
dataset.preprocess_data(
    preprocess_function_text, 
    lambda batch : preprocess_function_label(batch, LABEL2ID)
)
del LABEL2ID
print(dataset)

embedder = CustomEmbedder(config)
print(embedder)

classifier = CustomClassifier(config)
print(classifier)

model = CustomModel(config, embedder, classifier)

model.train(
    dataset.ds["train"],
    dataset.ds["validation"]
)
model.history.plot_all()
model.history.save_all("./testing-save")
model.clean()