from toolbox.Config import Config
from toolbox.CustomModel import CustomModel
from toolbox.CustomDataset import CustomDataset
from toolbox.CustomClassifier import CustomClassifier
from toolbox.CustomEmbedder import CustomEmbedder

import pandas as pd
from torch import DataLoader, zero_grad, Tensor

config = Config()
config.model_train_n_epoch = 5 # After first result analysis

config.history_foldersave = "./sklearn_save"

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


dataset.preprocess_data(
    preprocess_function_text, 
    lambda batch : preprocess_function_label(batch, config.dataset_label2id)
)


embedder = CustomEmbedder(config)
classifier = CustomClassifier(config)
model = CustomModel(config, embedder, classifier)

def callback_function_save_tensors(epoch : int, dataloader : DataLoader, 
        model : CustomModel, filename : str) -> None: 
    output_list : list[dict] = [] 
    with zero_grad():
        for id, batch in enumerate(dataloader) : 
            outputs : Tensor = model(batch["text"]) # shape(batch x config.embeddingmodel_dim)
            shape = outputs.shape
            for i in range(shape[0]) : 
                output_list.append({
                    "id" : f"{id}#{i}",
                    "text" : batch["text"][i],
                    "label" : batch["label"][i],
                    **{
                        f"dim_{j}" : outputs[i,j] for j in range(shape[1])
                    }
                })
    pd.DataFrame(output_list).to_csv(f"{filename}_{epoch}.csv", index=False)


model.train(dataset.ds["train"],dataset.ds["validation"],
    callback_function=callback_function_save_tensors,
    callback_parameters={
        "dataloader" : DataLoader(dataset.ds["train"], shuffle = True, 
                                  batch_size = config.model_train_batchsize),
        "model" : embedder,
        "filename" : "sklearn_save"
    })