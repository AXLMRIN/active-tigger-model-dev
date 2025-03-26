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
config.model_train_n_epoch = 1 # huge number of epoch for exploration purposes

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

test_loader = DataLoader(
    dataset.ds["test"], 
    batch_size=config.model_train_batchsize,
    shuffle = True
)

num_attempts = 2

for n_attempt in tqdm(range(num_attempts), 
                      desc = "Attempt", position = 0, leave = True):
    # pick random parameters
    config.model_train_classifier_learning_rate = choice(
        [1e-2 * (.5**i) for i in range(10)]
    )
    config.model_train_classifier_momentum = choice(
        [0.5 + 0.4 * i / 9 for i in range(10)]
    )
    config.model_train_classifier_weight_decay = choice(
        [0.001 + 0.099 * i / 19 for i in range(20)]
    )
    config.classifier_hiddenlayer_dim = choice([10, 50, 300])
    config.history_foldersave = f"./save_random_search/{n_attempt}"

    # train evaluate, save
    embedder = CustomEmbedder(config)
    classifier = CustomClassifier(config)
    model = CustomModel(config, embedder, classifier)
    model.train(dataset.ds["train"],dataset.ds["validation"])

    # evaluate on test data
    metrics : dict[str:float] = {"f1" : 0, "roc_auc" : 0, "accuracy" : 0}
    for batch in tqdm(test_loader, 
                      desc = "Testing loop", leave = False, position = 2):
        prediction_logits = model.predict(batch["text"], eval_grad = False)
        loss = model.loss_function_validation(
            prediction_logits.to(device = "cpu"), 
            batch["label"]
        )
        batch_metrics = model.evaluator(
            prediction_logits.to(device = "cpu"), 
            batch["label"]
        )
        metrics = {
            key : metrics[key] + batch_metrics[key] for key in metrics
        }
    metrics = {
        key : metrics[key] / len(test_loader) for key in metrics
    }
    model.history.append_metrics(-1, "test", metrics)
    
    model.history.save_all(config.history_foldersave)
    config.save()
    model.clean()
