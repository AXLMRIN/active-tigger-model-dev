from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.History import History
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder

from torch.utils.data import DataLoader

config = Config()
config.model_train_n_epoch = 5 # After first result analysis

config.history_foldersave = "./debug"
config.embeddingmodel_save_filename = f"{config.history_foldersave}/{config.embeddingmodel_save_filename}"
config.classifier_save_filename = f"{config.history_foldersave}/{config.classifier_save_filename}"

dataset = CustomDataset(config)
dataset.open_dataset()
dataset.find_labels()

def preprocess_function_label(batch, label2id): 
    return [label2id[label] for label in batch["label"]]
def preprocess_function_text(batch): 
    return [sentence.lower() for sentence in batch["text"]]


dataset.preprocess_data(
    preprocess_function_text, 
    lambda batch : preprocess_function_label(batch, config.dataset_label2id)
)


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