from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder

import os
from torch import no_grad, Tensor, concat, save
from torch.utils.data import DataLoader

def callback_function_save_tensors(epoch : int, dataloader : DataLoader, 
            model : CustomModel, filename : str) -> None: 
        output_list : Tensor|None = None
        idx = 0 

        with no_grad():
            for batch in dataloader: 
                outputs : Tensor = model(batch["text"]) # shape(batch x config.embeddingmodel_dim)
                indexes = Tensor([i for i in range(idx, idx + len(batch["text"]))]).to(device='cpu')
                outputs_cat = concat(
                    (
                        outputs.to(device='cpu'), 
                        batch["label"].to(device='cpu').unsqueeze(dim = 1),
                        indexes.to(device='cpu').unsqueeze(dim = 1)
                    ), 
                    axis = 1
                )
                idx += len(batch["text"])

                if output_list is None: output_list =outputs_cat
                else: output_list = concat((output_list, outputs_cat))
            
        save(outputs_cat, f"{filename}_{epoch}")


config = Config()
config.model_train_n_epoch = 5 # After first result analysis

config.history_foldersave = "./debug"
config.embeddingmodel_save_filename = f"{config.history_foldersave}/{config.embeddingmodel_save_filename}"
config.classifier_save_filename = f"{config.history_foldersave}/{config.classifier_save_filename}"

config.model_train_embedding_adam_parameters["lr"] = 5e-5
config.model_train_classifier_sgd_parameters["lr"] = 0.0003
config.classifier_hiddenlayer_dim = 10

dataset = CustomDataset(config)
dataset.open_dataset()
dataset.find_labels()

def preprocess_function_label(batch, label2id): 
    return [label2id[label] for label in batch["label"]]
def preprocess_function_text(batch): 
    return [sentence.lower() for sentence in batch["text"]]

# debug ---
dataset.ds["train"] = dataset.ds["train"].select(range(40))
dataset.ds["test"] = dataset.ds["test"].select(range(40))
dataset.ds["validation"] = dataset.ds["validation"].select(range(40))

dataset.preprocess_data(
    preprocess_function_text, 
    lambda batch : preprocess_function_label(batch, config.dataset_label2id)
)

if "sklearn_save" not in os.listdir("./") : 
     os.mkdir("./sklearn_save")

for model_name in [
    "google-bert/bert-base-uncased",
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large"
                  ]:
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
    model.train(dataset.ds["train"],dataset.ds["validation"],
        callback_function=callback_function_save_tensors,
        callback_parameters={
            "dataloader" : DataLoader(dataset.ds["train"], shuffle = True, 
                                    batch_size = config.model_train_batchsize),
            "model" : embedder,
            "filename" : f"./sklearn_save/{model_name_path}/epoch"
        })