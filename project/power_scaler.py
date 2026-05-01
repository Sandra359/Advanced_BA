import torch
import numpy as np

class PowerScaler:
    def __init__(self, X_train, y_train):
        tx = torch.from_numpy(X_train) if isinstance(X_train, np.ndarray) else X_train
        ty = torch.from_numpy(y_train) if isinstance(y_train, np.ndarray) else y_train
        
        self.x_mean = tx.mean(dim=(0, 1), keepdim=True)
        self.x_std  = tx.std(dim=(0, 1),  keepdim=True) + 1e-7
        self.y_mean = ty.mean()
        self.y_std  = ty.std() + 1e-7

    def scale_x(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.x_mean) / self.x_std

    def scale_y(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.y_mean) / self.y_std

    def inverse_y(self, tensor):
        """Undoes standardisation. """
        result = tensor * float(self.y_std) + float(self.y_mean)
        return result

    def inverse_x(self, tensor):
        return (tensor * self.x_std) + self.x_mean