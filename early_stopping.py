import numpy as np
import torch

import math

# Notes:
# Adapted from [1].

# References:
# [1] https://github.com/Bjarten/early-stopping-pytorch
# [2] https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

class EarlyStopping:
    """Stop training when validation loss does not improve."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        # >>>>
        self.best_epoch = None

    ### def __call__(self, val_loss, model):
    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            ### print('--> First epoch...')
            if self.verbose:
                print('--> First epoch: saving initial model')

            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # >>>>
            self.best_epoch = epoch

            return

        # No improvement
        if score < self.best_score + self.delta or math.isnan(val_loss):
            # Score decreased. Loss increased.
            ### print('--> Loss increased... Score decreased')
            self.counter += 1
            if self.verbose:
                print(f'--> No improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True

        # Improvement
        else:
            ### print('--> Loss decreased or stayed the same... Score increased or stayed the same...')
            if self.verbose:
                print('--> Improvement detected: saving model')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            # >>>>
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}).  Saving model to {self.path}.')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss