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
config.model_train_n_epoch = 5 # After first result analysis

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

num_attempts = 1

for n_attempt in tqdm(range(num_attempts), 
                      desc = "Attempt", position = 0, leave = True):
    # pick random parameters
    # Classifier
    config.model_train_classifier_learning_rate = choice(
        [0.01, 0.005, 1e-6]
    )

    config.model_train_classifier_momentum = choice(
        [0.45, 0.50, 0.55, 0.60, 0.65]
    )

    config.model_train_classifier_weight_decay = choice(
        [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    )

    config.classifier_hiddenlayer_dim = choice(
        [50, 100, 150, 200, 250]
    )
    # Embedding
    config.model_train_embedding_learning_rate = choice(
        [200e-6, 300e-6, 400e-6, 50e-6]
    )

    config.model_train_embedding_weight_decay = choice(
        [0.001, 0.075, 0.05, 0.1]
    )

    config.embeddingmodel_output = choice(
        ["pooler_output", "last_hidden_state"]
    )

    config.history_foldersave = f"./save_random_search/{n_attempt}"
    config.embeddingmodel_save_filename = (f"{config.history_foldersave}/"
                                         f"{config.embeddingmodel_save_filename}")
    config.classifier_save_filename = (f"{config.history_foldersave}/"
                                    f"{config.classifier_save_filename}")
    # train evaluate, save
    embedder = CustomEmbedder(config)
    classifier = CustomClassifier(config)
    model = CustomModel(config, embedder, classifier)
    model.train(dataset.ds["train"],dataset.ds["validation"])

    # evaluate on test data
    model.load_best()
    model.embedder.eval()
    model.classifier.eval()
    metrics : dict[str:float] = {"f1" : 0, "roc_auc" : 0, "accuracy" : 0}

    for batch in tqdm(test_loader, 
                      desc = "Testing loop", leave = False, position = 1):
        prediction_logits = model.predict(
            batch["text"], 
            eval_grad = False,
            use_best = True
        )
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
