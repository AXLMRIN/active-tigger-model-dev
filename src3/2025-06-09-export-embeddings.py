import gc
import os
from torch import Tensor, no_grad, cat, save
from torch.cuda import is_available as cuda_available
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset, Dataset, DatasetDict

def embed_all(foldername : str, dataset_folder : str, epoch : int) -> Tensor:
    device = "cuda"
    model = None
    ds = None
    embeddings = None
    labels = None
    try : 
        model = AutoModelForSequenceClassification.from_pretrained(foldername).\
            to(device = device)

        ds = load_dataset("arrow", 
                data_files={
                    'train': f'{dataset_folder}/train/data-00000-of-00001.arrow',
                    'eval': f'{dataset_folder}/eval/data-00000-of-00001.arrow',
                    'test': f'{dataset_folder}/test/data-00000-of-00001.arrow',
                }
            )

        if f"epoch_{epoch}" not in os.listdir(dataset_folder):
            os.mkdir(f"{dataset_folder}/epoch_{epoch}")

        with no_grad():
            embeddings = None
            labels = None
            for batch in DataLoader(ds["train"], batch_size=32, shuffle=True) : 
                batch_labels = Tensor([vec.tolist() for vec in batch["labels"]]).\
                                int().T.to(device="cpu")
                output = model.base_model(**{
                    key : Tensor([vec.tolist() for vec in batch[key]]).T.\
                            int().to(device=model.device)
                    for key in batch if key!= "labels"
                }).last_hidden_state[:,0,:].squeeze()

                if embeddings is None:
                    embeddings = output
                else :
                    embeddings = cat((embeddings,output), axis = 0)

                if labels is None:
                    labels = batch_labels
                else :
                    labels = cat((labels,batch_labels), axis = 0)
                    
            for batch in DataLoader(ds["eval"], batch_size=32, shuffle=True) : 
                batch_labels = Tensor([vec.tolist() for vec in batch["labels"]]).\
                                int().T.to(device="cpu")
                output = model.base_model(**{
                    key : Tensor([vec.tolist() for vec in batch[key]]).T.\
                            int().to(device=model.device)
                    for key in batch if key!= "labels"
                }).last_hidden_state[:,0,:].squeeze()
                
                if embeddings is None:
                    embeddings = output
                else :
                    embeddings = cat((embeddings,output), axis = 0)

                if labels is None:
                    labels = batch_labels
                else :
                    labels = cat((labels,batch_labels), axis = 0)

            print(embeddings.shape, labels.shape)
            save(embeddings,f"{dataset_folder}/epoch_{epoch}/train_embedded.pt")
            save(labels,f"{dataset_folder}/epoch_{epoch}/train_labels.pt")

            embeddings = None
            labels = None
            for batch in DataLoader(ds["test"], batch_size=32, shuffle=True) : 
                batch_labels = Tensor([vec.tolist() for vec in batch["labels"]]).\
                                int().T.to(device="cpu")
                output = model.base_model(**{
                    key : Tensor([vec.tolist() for vec in batch[key]]).T.\
                            int().to(device=model.device)
                    for key in batch if key!= "labels"
                }).last_hidden_state[:,0,:].squeeze()
                
                if embeddings is None:
                    embeddings = output
                else :
                    embeddings = cat((embeddings,output), axis = 0)
                
                if labels is None:
                    labels = batch_labels
                else :
                    labels = cat((labels,batch_labels), axis = 0)

            print(embeddings.shape, labels.shape)
            save(embeddings,f"{dataset_folder}/epoch_{epoch}/test_embedded.pt")
            save(labels,f"{dataset_folder}/epoch_{epoch}/test_labels.pt")

    finally:
        del model, ds, embeddings, labels, batch_labels, output
        empty_cache()
        synchronize()
        ipc_collect()
        gc.collect()
    return "Done"

root = "./src3/319_models/2025-06-09"
model_name = "answerdotai/ModernBERT-base"
learning_rate = "0.0001"

for learning_rate in ["0.0001", "1e-05", "5e-05", "1e-06"]:
    print("=" * 100)
    print(learning_rate)

    path = f"{root}-{model_name}-{learning_rate}"
    checkpoints = os.listdir(path)
    checkpoints = sorted(checkpoints, key = lambda x : int(x.split("-")[-1]))

    # Embed for untrained model
    _ = embed_all(model_name,f"{path}-data",0)

    # Embed for trained models
    for i, checkpoint in enumerate(checkpoints):
        print(i + 1, "\t", checkpoint)
        _ = embed_all(
            f"{path}/{checkpoint}",
            f"{path}-data",
            i + 1
        )
        print("")