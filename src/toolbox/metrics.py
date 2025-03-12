from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch import where as torch_where

def threshold(probabilities : Tensor, thresh_value : float = 0.4) -> Tensor :
    """Everything happens on the cpu
    probabilities is already on the cpu
    """
    return torch_where(
        probabilities > thresh_value,
        Tensor([1.0]),
        Tensor([0.0])
    ).to(dtype = bool, device = "cpu", non_blocking=True)

def classifier_metrics(target : Tensor, probabilities : Tensor, 
                       thresh_value : float = 0.4) -> dict[str:float]:
    """Everything happens on the cpu
    probabilities = probabilities.detach().cpu()
    target is already on the cpu and of dtype bool
    """
    target.to(device = "cpu", dtype = bool, non_blocking=True) # just to make sure but shouldn't be necessary
    y_pred = threshold(probabilities, thresh_value) # bool, on cpu
    
    return {
        'f1': f1_score(y_true = target, y_pred = y_pred, average='micro'),
        'roc_auc': roc_auc_score(target, y_pred, average = 'micro').item(),
        'accuracy': accuracy_score(target, y_pred)
    }