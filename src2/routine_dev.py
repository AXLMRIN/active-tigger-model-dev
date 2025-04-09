from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.History import History
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder
from copy import deepcopy

from torch.utils.data import DataLoader
from tqdm import tqdm
from random import choice

config = Config()
config.model_train_n_epoch = 2 # After first result analysis

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

# debug
config.model_train_batchsize = 8
dataset.ds["train"] = dataset.ds["train"].select(range(80))
dataset.ds["validation"] = dataset.ds["validation"].select(range(80))
dataset.ds["test"] = dataset.ds["test"].select(range(80))


embedder = CustomEmbedder(config)
classifier = CustomClassifier(config)
model = CustomModel(config, embedder, classifier)
model.train(dataset.ds["train"],dataset.ds["validation"])

test_loader = DataLoader(
    dataset.ds["test"], 
    batch_size=config.model_train_batchsize,
    shuffle = True
)
model.test_loop(test_loader)

model.history.save_all(config.history_foldersave)
config.save()
model.clean()