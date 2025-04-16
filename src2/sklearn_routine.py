from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder
from toolbox.CustomLogger import CustomLogger

import os
from torch import no_grad, Tensor, concat, save
from torch.utils.data import DataLoader

def callback_function_save_tensors(epoch : int, 
            dataloader_train : DataLoader, 
            dataloader_valid : DataLoader, 
            dataloader_test : DataLoader,
            model : CustomModel, filename : str) -> None: 
        with no_grad():
            # Train dataset ====================================================
            full_output : Tensor|None = None
            idx = 0 
            for batch in dataloader_train: 
                embeddings : Tensor = model(batch["text"]) # shape(batch x config.embeddingmodel_dim)
                indexes = Tensor([i for i in range(idx, idx + len(batch["text"]))]).\
                            to(device='cpu')
                output_batch = concat(
                    (
                        indexes.to(device='cpu').unsqueeze(dim = 1),
                        embeddings.to(device='cpu'), 
                        batch["label"].to(device='cpu').unsqueeze(dim = 1)
                    ), 
                    axis = 1
                )
                idx += len(batch["text"])

                if full_output is None: 
                    full_output = output_batch
                else: 
                    full_output = concat((full_output, output_batch))
            print(full_output.shape) # TODELETE
            # Validation dataset ===============================================
            # concatenate the train and validation
            for batch in dataloader_valid: 
                embeddings : Tensor = model(batch["text"]) # shape(batch x config.embeddingmodel_dim)
                indexes = Tensor([i for i in range(idx, idx + len(batch["text"]))]).\
                            to(device='cpu')
                output_batch = concat(
                    (
                        indexes.to(device='cpu').unsqueeze(dim = 1),
                        embeddings.to(device='cpu'), 
                        batch["label"].to(device='cpu').unsqueeze(dim = 1)
                    ), 
                    axis = 1
                )
                idx += len(batch["text"])

                if full_output is None: 
                    full_output = output_batch
                else: 
                    full_output = concat((full_output, output_batch))
            print(full_output.shape) # TODELETE
            save(full_output, f"{filename}_{epoch}_train.pt")
            # Test dataset =====================================================
            full_output : Tensor|None = None
            idx = 0 
            for batch in dataloader_valid: 
                embeddings : Tensor = model(batch["text"]) # shape(batch x config.embeddingmodel_dim)
                indexes = Tensor([i for i in range(idx, idx + len(batch["text"]))]).\
                            to(device='cpu')
                output_batch = concat(
                    (
                        indexes.to(device='cpu').unsqueeze(dim = 1),
                        embeddings.to(device='cpu'), 
                        batch["label"].to(device='cpu').unsqueeze(dim = 1)
                    ), 
                    axis = 1
                )
                idx += len(batch["text"])

                if full_output is None: 
                    full_output = output_batch
                else: 
                    full_output = concat((full_output, output_batch))
            print(full_output.shape) # TODELETE
            save(full_output, f"{filename}_{epoch}_test.pt")


config = Config()
config.model_train_n_epoch = 5 # After first result analysis

config.history_foldersave = "./debug"
config.embeddingmodel_save_filename = f"{config.history_foldersave}/{config.embeddingmodel_save_filename}"
config.classifier_save_filename = f"{config.history_foldersave}/{config.classifier_save_filename}"

config.model_train_embedding_adam_parameters["lr"] = 5e-5
config.model_train_classifier_sgd_parameters["lr"] = 0.0003
config.classifier_hiddenlayer_dim = 10
config.model_save_best_model = False

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

if "sklearn_save" not in os.listdir("./") : 
     os.mkdir("./sklearn_save")

for model_name in [
    "google-bert/bert-base-uncased",
    "answerdotai/ModernBERT-base",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
    "answerdotai/ModernBERT-large",
                  ]:
    print(model_name)
    model_name_path = model_name.replace("/","_")
    if model_name_path not in os.listdir("./sklearn_save") : 
         os.mkdir(f"./sklearn_save/{model_name_path}")

    config.embeddingmodel_name = model_name
    embedder = CustomEmbedder(config)
    classifier = CustomClassifier(config)
    model = CustomModel(config, embedder, classifier)

    

    callback_function_save_tensors(-1,
        DataLoader(
            dataset=dataset.ds["train"], 
            shuffle = True, 
            batch_size = config.model_train_batchsize
        ),
        embedder,
        f"./sklearn_save/{model_name_path}/epoch"
    )

    if not(model_name.endswith("large")):
        model.train(dataset.ds["train"],dataset.ds["validation"],
            callback_function=callback_function_save_tensors,
            callback_parameters={
                "dataloader" : DataLoader(dataset.ds["train"], shuffle = True, 
                                        batch_size = config.model_train_batchsize),
                "model" : embedder,
                "filename" : f"./sklearn_save/{model_name_path}/epoch"
            })
    model.clean()

CustomLogger().notify_when_done()