import numpy as np
from torch import load
from sklearn.utils import resample

class DATA:
    def __init__(self, foldername_root : str, epoch : int, n_samples : int = 500):
        self.X_train = load(f"./{foldername_root}/epoch_{epoch}/train_embedded.pt",
            weights_only=True).cpu().numpy()
        labels = load(f"./{foldername_root}/epoch_{epoch}/train_labels.pt",
            weights_only=True).cpu().numpy()
        self.y_train = [np.argmax(row).item() for row in labels]
        self.X_train, self.y_train = resample(self.X_train,self.y_train, n_samples=n_samples)

        self.X_test = load(f"./{foldername_root}/epoch_{epoch}/test_embedded.pt",
            weights_only=True).cpu().numpy()
        labels = load(f"./{foldername_root}/epoch_{epoch}/test_labels.pt",
            weights_only=True).cpu().numpy()
        self.y_test = [np.argmax(row).item() for row in labels]
        self.X_test, self.y_test = resample(self.X_test,self.y_test)