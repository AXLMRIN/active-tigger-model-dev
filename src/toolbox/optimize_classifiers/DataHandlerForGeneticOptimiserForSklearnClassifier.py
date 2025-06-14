
# IMPORTS ######################################################################
import numpy as np
from torch import load
from sklearn.utils import resample
from .. import ROOT_MODELS
# SCRIPTS ######################################################################
class DataHandlerForGeneticOptimiserForSklearnClassifier:
    """
    """
    def __init__(self, foldername) -> None:
        """
        """
        self.X_train : np.ndarray = load(f"{ROOT_MODELS}/{foldername}/train_embedded.pt",
                            weights_only=True).cpu().numpy
        self.y_train : np.ndarray = load(f"{ROOT_MODELS}/{foldername}/train_labels.pt",
                            weights_only=True).cpu().numpy
        self.X_test : np.ndarray = load(f"{ROOT_MODELS}/{foldername}/test_embedded.pt",
                            weights_only=True).cpu().numpy
        self.y_test : np.ndarray = load(f"{ROOT_MODELS}/{foldername}/test_labels.pt",
                            weights_only=True).cpu().numpy