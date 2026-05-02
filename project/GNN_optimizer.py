import torch


class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model checkpoint to disk so weights are never lost.

    Args:
        patience  : epochs to wait after last improvement before stopping
        min_delta : minimum decrease in val loss that counts as an improvement
        path      : filepath for the best-model checkpoint
        verbose   : print a line each epoch summarising progress
    """
    def __init__(self, patience=15, min_delta=1e-4,
                 path="best_stgnn.pt", verbose=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.verbose    = verbose

        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"    ✓ val loss improved → {val_loss:.4f}  (checkpoint saved)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stop = True

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path, weights_only=True))
        print(f"  Loaded best weights from epoch {self.best_epoch} "
              f"(val loss: {self.best_loss:.4f})")
        return model