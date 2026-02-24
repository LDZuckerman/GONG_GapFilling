import numpy as np
import cv2
import scipy.ndimage as sndi
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import json
try:
    import models, loss_funcs, plot_utils
except ModuleNotFoundError:
    from utils import models, loss_funcs, plot_utils


def train_net(loader, model, loss_name, optimizer, device, save_examples=True, save_dir=None, epoch=None):
    '''
    Train supervised model for one epoch
    '''

    # Train on each batch in train loader
    k = 0
    batch_size = next(iter(loader))[0].shape[0]
    for data, targets in loader: #batch_idx, (data, targets) in enumerate(loop):
        
        print(f'  --> Batch {k}', end='\r', flush=True); k += 1
        
        # Set data to be on correct device
        data = data.to(device)
        targets = targets.float().to(device)
        
        # Forward
        predictions = model(data)

        # Loss
        if torch.isnan(predictions).any() or not torch.isfinite(predictions).all():
            raise ValueError('preds become NaN or inf') 
        loss_func = getattr(loss_funcs, loss_name) # e.g. nn.MSELoss()
        if loss_name == 'MSE':
            loss = loss_func(predictions, targets) 
        elif loss_name == 'Selected_MSE':
            loss = loss_func(predictions, targets, data) 
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 
        
        
    # Plot examples from last batch 
    if save_examples: 
        plot_utils.plot_epoch_examples(data, torch.tensor(targets), torch.tensor(predictions), save_dir, epoch) # data, targets will be those loaded from last batch 

        
    return loss


def save_model_results(val_loader, save_dir, model, device='cpu'):
    '''
    Run each validation obs through model, save results
    '''
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        if torch.is_tensor(y):
            y = y.cpu().detach().numpy()
        preds = model(X).cpu().detach().numpy()
        for j in range(np.shape(preds)[0]):
            np.save(f'{save_dir}/x_{i}', X[j].cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1       