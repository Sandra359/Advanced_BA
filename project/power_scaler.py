import torch
import numpy as np

class PowerScaler:
    def __init__(self, X_train, y_train):
        # Convert inputs to torch tensors once for stat calculation
        tx = torch.from_numpy(X_train) if isinstance(X_train, np.ndarray) else X_train
        ty = torch.from_numpy(y_train) if isinstance(y_train, np.ndarray) else y_train
        
        # X stats: Shape (1, 1, 1, Num_Features) 
        # This treats each feature type globally across all time and nodes
        self.x_mean = tx.mean(dim=(0, 1, 2), keepdim=True)
        self.x_std  = tx.std(dim=(0, 1, 2),  keepdim=True) + 1e-7
        
        # y stats: Scalar values for the target
        self.y_mean = ty.mean()
        self.y_std  = ty.std() + 1e-7

    def scale_x(self, data):
        """Converts numpy/tensor to scaled torch tensor"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.x_mean) / self.x_std

    def scale_y(self, data):
        """Scales target values"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.y_mean) / self.y_std

    def inverse_x(self, tensor):
        """Returns X to original units (e.g., for interpreting inputs)"""
        return (tensor * self.x_std) + self.x_mean

    def inverse_y(self, tensor):
        y_std  = float(self.y_std)
        y_mean = float(self.y_mean)
        return (tensor * y_std) + y_mean