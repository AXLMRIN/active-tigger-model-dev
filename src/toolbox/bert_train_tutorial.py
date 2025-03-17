from datasets.formatting.formatting import LazyBatch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn import Sigmoid
from transformers import EvalPrediction
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

def preprocess_data(batch_of_rows : LazyBatch, tokenizer : BertTokenizerFast,
         labels : list[int], label2id : dict[str:int],
         sentence_column : str = "Tweet",) -> BatchEncoding:
    # Takes in a batch of rows (as a : LazyBatch ~ dataframe ish) 
    
    # collect the text and tokenize it 
    text = batch_of_rows[sentence_column]
    encoding : BatchEncoding = tokenizer(
        text, padding = "max_length", truncation = True, max_length = 128 
    )
    # Create a mattrix collecting all the metadata (emotions associated to the 
    # tweet)
    labels_matrix = np.zeros((len(text), len(labels)))
    for label in label2id:
        labels_matrix[:,label2id[label]] = batch_of_rows[label]

    # Associate the metadata to the encodings
    encoding["labels"] = labels_matrix.tolist()
    return encoding

def get_labels(example_labels : Tensor, id2label : dict[int:str]) -> list[str]:
    return [id2label[idx] 
            for idx,label in enumerate(example_labels) if label == 1]


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/ 
def multi_label_metrics(results_matrix, labels : Tensor, threshold : float = 0.5
                        ) -> dict:
    '''Taking a results matrix (batch_size x num_labels), the function (with a 
    threshold) associates labels to the results => y_pred
    From this y_pred matrix, evaluate the f1_micro, roc_auc and accuracy metrics
    '''
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = Sigmoid()
    probs = sigmoid(Tensor(results_matrix))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    return {'f1': f1_micro_average,
             'roc_auc': roc_auc,
             'accuracy': accuracy}

def compute_metrics(model_output: EvalPrediction):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions

    metrics = multi_label_metrics(results_matrix=results_matrix, 
        labels=model_output.label_ids)
    return metrics