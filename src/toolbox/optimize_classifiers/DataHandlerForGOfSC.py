
# IMPORTS ######################################################################
import numpy as np
from torch import load
from sklearn.utils import resample
from .. import ROOT_MODELS
# SCRIPTS ######################################################################
class DataHandlerForGOfSC:
    """
    """
    def __init__(self, foldername : str, n_samples : int|None = None) -> None:
        """
        """
        if foldername.startswith("./") : 
            root = foldername
        else :
            root = f"{ROOT_MODELS}/{foldername}"

        self.X_train : np.ndarray = load(f"{root}/train_embeddings.pt",
                            weights_only=True).cpu().numpy()
        labels : np.ndarray = load(f"{root}/train_labels.pt",
                            weights_only=True).cpu().numpy()
        self.y_train = [np.argmax(row).item() for row in labels]

        # Resample
        if n_samples is not None : 
            self.X_train, self.y_train = resample(self.X_train,self.y_train, 
                                                  n_samples=n_samples)
            
        self.X_test : np.ndarray = load(f"{root}/test_embeddings.pt",
                            weights_only=True).cpu().numpy()
        labels : np.ndarray = load(f"{root}/test_labels.pt",
                            weights_only=True).cpu().numpy()
        self.y_test = [np.argmax(row).item() for row in labels]