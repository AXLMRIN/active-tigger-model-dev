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


lr_classifier_to_test = [1e-2,1e-3,1e-4,1e-5]
lr_embedder_to_test = [1e-2,1e-3,1e-4,1e-5]
for iLRC in tqdm(range(len(lr_classifier_to_test)),desc = "lr_classifier_to_test", position= 2):
    for iLRE in tqdm(range(len(lr_embedder_to_test)), desc = "lr_embedder_to_test", position = 3):
        config = Config()
        config.model_train_n_epoch = 5 # After first result analysis

        config.history_foldersave = "./LRE_{iLRE}_LRC_{iLRC}"
        config.embeddingmodel_save_filename = f"{config.history_foldersave}/{config.embeddingmodel_save_filename}"
        config.classifier_save_filename = f"{config.history_foldersave}/{config.classifier_save_filename}"

        config.model_train_embedding_adam_parameters["lr"] = lr_embedder_to_test[iLRE]
        config.model_train_classifier_sgd_parameters["lr"] = lr_classifier_to_test[iLRC]

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