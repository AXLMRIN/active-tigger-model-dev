from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import TrainOutput
from transformer_functions import compute_metrics
from torch.cuda import is_available as cuda_available

class dataset:
    def __init__(self, csv_filename : str, col_text : str, col_label : str) -> None:
        self.data : pd.DataFrame = pd.\
            read_csv(csv_filename).\
            rename({col_text : "text",col_label : "label"}, axis = 1).\
            loc[:, ["text", "label"]].\
            sample(frac = 1) # shuffle all

        # Fetch labels
        self.label2id = {}
        self.id2label = {}
        self.label = []
        for id, (label, _) in enumerate(self.data.groupby("label")):
            self.label2id[label] = id
            self.id2label[id] = label
            self.label.append(label)
        self.n_labels = len(self.label)

        # Splitting parameters
        n_max : int = min(self.data.groupby("label").size())
        self.N_train : int = int(0.7 * n_max)
        self.N_eval : int = int(0.15 * n_max)
        self.N_test : int = n_max - self.N_train - self.N_eval

    def __repr__(self) -> str:
        return (f'Dataset : \n'
                f'{self.data.groupby("label").size()}\n'
                f'\n'
                f'|{"N_train":>15} | {self.N_train:<5} |\n'
                f'|{"N_eval":>15} | {self.N_eval:<5} |\n'
                f'|{"N_test":>15} | {self.N_test:<5} |')
    
    def concat_shuffle(self,start,finish) -> pd.DataFrame :
        grouped = self.data.groupby("label")
        return pd.concat([
            grouped.get_group(label)[start:finish]
            for label in self.label2id
            ]).sample(frac = 1) # shuffle

    def collect(self) -> DatasetDict : 
        return DatasetDict({
            "train" : Dataset.from_pandas(
                self.concat_shuffle(0,self.N_train)),
            "eval" : Dataset.from_pandas(
                self.concat_shuffle(self.N_train, self.N_train + self.N_eval)),
            "test" : Dataset.from_pandas(
                self.concat_shuffle(self.N_train + self.N_eval,
                                self.N_train + self.N_eval + self.N_test))
        })

class transformer:
    def __init__(self, ds : dataset, model_name : str, **kwargs) -> None:
        self.ds : dataset = ds
        self.ds_data : DatasetDict = ds.collect()
        encoded_dataset = None

        self.model_name : str = model_name

        self.device : str = "cuda" if cuda_available() else "cpu"
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.\
                            from_pretrained(self.model_name)
        model_parameters = {
            "problem_type" : "multi_label_classification",
            "num_labels" : self.ds.n_labels, 
            "id2label" : self.ds.id2label,
            "label2id" : self.ds.label2id
        }
        self.model = AutoModelForSequenceClassification.\
                        from_pretrained(self.model_name, **model_parameters).\
                        to(device = self.device)

        # Parameters 
        tokenizer_max_length = 128 if 'tokenizer_max_length' not in kwargs else kwargs['tokenizer_max_length']
        self.tokenizer_p = {
            'padding' : 'max_length',
            'truncation' : True,
            'max_length' : min(
                self.model.config.max_position_embeddings,
                tokenizer_max_length
            )
        }

        total_per_batch = 64
        batch_size_device = 8
        self.training_args = TrainingArguments(
            num_train_epochs=5,
            # bf16=True,
            # Hyperparameters
            learning_rate=2e-5,
            # optim_args = {}
            weight_decay=0.01,
            warmup_ratio = 0.1,
            # Second order hyperparameters
            per_device_train_batch_size = batch_size_device,
            per_device_eval_batch_size = batch_size_device,
            gradient_accumulation_steps = int(total_per_batch/ batch_size_device),
            optim = "adamw_torch",
            # Metrics
            metric_for_best_model="f1",
            # Pipe
            output_dir = "2025-04-23-bert-GA",
            overwrite_output_dir=True,
            eval_strategy = "epoch",
            logging_strategy = "epoch",
            save_strategy = "epoch",
            torch_empty_cache_steps = int(
                self.ds.n_labels * self.ds.N_train / batch_size_device),
            load_best_model_at_end=True,
            save_total_limit = 1,

            disable_tqdm = True
        )
            
    
    def preprocess(self, preprocessing_function) -> None : 
        self.ds_data = self.ds_data.map(preprocessing_function,
                             batched=True,batch_size=64)
    
    def encode(self):
        self.encoded_dataset = DatasetDict()

        for ds_name in ["train","eval", "test"] : 
            list_of_rows : list[dict] = []
            for row in self.ds_data[ds_name] :
                # row : {'text' : str, 'label' : str} 
                tokens : BatchEncoding = self.tokenizer(
                    row["text"], **self.tokenizer_p)
                
                list_of_rows.append({
                    "input_ids" : tokens.input_ids,
                    "token_type_ids" : tokens.token_type_ids,
                    "attention_mask" : tokens.attention_mask,
                    "labels" : [float(id == self.ds.label2id[row["label"]])
                                for id in range(self.ds.n_labels)]
                })
            self.encoded_dataset[ds_name] = Dataset.from_list(list_of_rows)
    
    def debug_mode(self):
        if self.encoded_dataset is None: 
            print("the dataset is not yet encoded, use transformer.encode first")
        else : 
            # Debug
            self.encoded_dataset["train"] = self.encoded_dataset["train"].\
                                                select(range(20))
            self.encoded_dataset["eval"] = self.encoded_dataset["eval"].\
                                                select(range(20))
        
    def train(self) -> TrainOutput:
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.encoded_dataset["train"],
            eval_dataset = self.encoded_dataset["eval"],
            compute_metrics = compute_metrics,
            # optimizers = 
        )
        return trainer.train()


import os
from email.message import EmailMessage
import ssl
import smtplib
class CustomLogger:
    def __init__(self):
        self.name = ""

    def notify_when_done(self, message : str = '') : 
        """send an email when finished"""
        subj = "Onyxia run — done"
        body = ("https://projet-datalab-axel-morin-135428-0.lab.groupe-genes.fr/lab?"
                f"\n{message}")
        em = EmailMessage()
        em["From"] = os.environ["EMAIL_FROM"]
        em["To"] = os.environ["EMAIL_TO"]
        em["Subject"] = subj
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp : 
            print(smtp.login(os.environ["EMAIL_FROM"], os.environ["EMAIL_FROM_PWD"]))
            print(smtp.sendmail(
                os.environ["EMAIL_FROM"],
                os.environ["EMAIL_TO"], 
                em.as_string())
            )

    def __str__(self) -> str:
        return (
            "Custom Logger object"
        )