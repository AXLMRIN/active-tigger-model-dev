from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.CustomLogger import CustomLogger
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = CustomLogger()

lr_classifier_to_test = [1e-3, 7e-4, 5e-5, 3e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5]
lr_embedder_to_test = [1e-5, 3e-5, 5e-5, 7e-5, 5e-6, 1e-6]
dim = [5, 10, 50, 150]
for iLRC in tqdm(range(len(lr_classifier_to_test)),desc = "lr_classifier_to_test", position= 4):
    for iLRE in tqdm(range(len(lr_embedder_to_test)), desc = "lr_embedder_to_test", position = 5):
        for iDIM in tqdm(range(len(dim)), desc = "dim", position = 6):
            config = Config()
            config.model_train_n_epoch = 5 # After first result analysis

            config.history_foldersave = f"./2025-04-09-grid-search/LRE_{iLRE}_LRC_{iLRC}_DIM_{iDIM}"
            config.embeddingmodel_save_filename = f"{config.history_foldersave}/{config.embeddingmodel_save_filename}"
            config.classifier_save_filename = f"{config.history_foldersave}/{config.classifier_save_filename}"

            config.model_train_embedding_adam_parameters["lr"] = lr_embedder_to_test[iLRE]
            config.model_train_classifier_sgd_parameters["lr"] = lr_classifier_to_test[iLRC]
            config.classifier_hiddenlayer_dim = dim[iDIM]
            
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

logger.notify_when_done()