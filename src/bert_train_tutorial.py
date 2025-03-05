from datasets import load_dataset   
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from toolbox.bert_train_tutorial import (
    preprocess_data, compute_metrics
)
from time import time

if __name__ == "__main__" : 
    custom_bert_model_name = "models/2025-03-04-bert-base-uncased"
    model_name = "bert-base-uncased" # FIXME
    dataset = load_dataset("csv", data_files = {
        "train" : "data/bert_train_tuto/train.csv",
        "test" : "data/bert_train_tuto/test.csv",
        "validation" : "data/bert_train_tuto/validation.csv"
    })

    labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded_dataset = dataset.map(lambda batch_of_rows : preprocess_data(
        batch_of_rows,tokenizer, labels, label2id), batched = True, 
        remove_columns=dataset["train"].column_names)

    encoder_classifier = AutoModelForSequenceClassification.from_pretrained(
        custom_bert_model_name,
        problem_type = "multi_label_classification", num_labels = len(labels),
        id2label = id2label, label2id = label2id)
    
    batch_size = 8
    metric_name = "f1"

    training_args = TrainingArguments(
        output_dir = "2025-03-04-classifieur_entraine",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        #push_to_hub=True,
    )

    trainer = Trainer(encoder_classifier, training_args,
                  train_dataset = encoded_dataset["train"].select(range(0,20)),
                  eval_dataset = encoded_dataset["validation"].select(range(0,10)),
                  tokenizer = tokenizer,
                  compute_metrics = compute_metrics)
    
    t1 = time()
    trainer.train()
    print(f"{time()-t1:2f} s to train")
    del t1