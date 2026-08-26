import numpy as np
import cv2
import scipy.ndimage as sndi
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import json, pickle
try:
    import models, loss_funcs, plot_utils
except ModuleNotFoundError:
    from utils import models, loss_funcs, plot_utils
    

def train_net(loader, model, loss_name, ctr_wgt, optimizer, device, save_examples=True, save_dir=None, epoch=None):
    '''
    Train supervised model for one epoch
    '''

    # Train on each batch in train loader
    k = 0
    batch_size = next(iter(loader))[0].shape[0]
    for data, targets in loader: #batch_idx, (data, targets) in enumerate(loop):
          
        # Set data to be on correct device
        data = data.float().to(device)
        targets = targets.float().to(device)
        
        # Forward
        predictions = model(data)

        # Loss
        if torch.isnan(predictions).any():
            raise ValueError(f'Batch {k}: Predictions have become NaN. Loss on last batch was {loss}') 
        if not torch.isfinite(predictions).all():
            raise ValueError(f'Batch {k}: Predictions have become infinite. Loss on last batch was {loss}') 
        loss_func = getattr(loss_funcs, loss_name) # e.g. nn.MSELoss()
        if loss_name == 'MSE':
            loss = loss_func(predictions, targets) 
        elif loss_name in ['Selected_MSE', 'Tuned_MSE', 'Selected_SSI', 'Selected_MSE_1D']:
            loss = loss_func(predictions, targets, data) 
        elif loss_name in ['Regional_MSE', 'Gradational_MSE']:
            loss = loss_func(predictions, targets, data, ctr_wgt) 
        else:
            raise ValueError(f'Loss "{loss_name}" not recognized')
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

    # Plot examples from last batch 
    if save_examples and data.dim() == 4: 
        plot_utils.plot_epoch_examples(data, torch.tensor(targets), torch.tensor(predictions), save_dir, epoch) # data, targets will be those loaded from last batch 
 
    return loss # this is loss from most recent train batch


def get_model(d, xs0, device):
    '''
    Helper function to return model 
    '''
    
    if d['model_name'] == 'SimpleCNN':
        model = models.SimpleCNN(len_set=xs0.shape[1], k_size=d['kernel_size'], padding_mode=d['padding_mode']).to(device)
    elif d['model_name'] == 'SimpleRNN':
        model = models.SimpleRNN(n_pix=xs0.shape[2], hidden_channels=[16, 32], num_layers=2).to(device)
    elif d['model_name'] == 'UNet':
        convblock_depth = 2 if 'convblock_depth' not in d.keys() else d['convblock_depth']
        model = models.UNet(len_set=xs0.shape[1], convblock_depth=convblock_depth).to(device)
    elif d['model_name'] == 'UNet2':
        convblock_depth = 2 if 'convblock_depth' not in d.keys() else d['convblock_depth']
        model = models.UNet2(len_set=xs0.shape[1], convblock_depth=convblock_depth).to(device)
    elif d['model_name'] == 'BRITS_mod':
        model = models.BRITS_mod(rnn_hid_size=d['rnn_hid_size'], seq_len=d['seq_len']).to(device)
    elif d['model_name'] == 'CNN1D':
        model = models.CNN1D(len_set=xs0.shape[1]).to(device)
    elif d['model_name'] == 'UNet1D':
        model = models.UNet1D().to(device)
        
    return model


def save_model_results(val_loader, file_names, save_dir, model, device='cpu', prnt=True):
    '''
    Run each validation obs through model, save results
        NOTE: Expects un-normalized data in loader, so that we can save images with their original value ranges
    '''
    
    # Make dir
    if prnt: print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: 
        os.mkdir(save_dir)
        
    # Load file names
    if file_names != None:
        starttimes = [f[f.find('/')+1:f.find('_to')] for f in file_names] 
        starttimes_dict = {}
    
    # Loop over val loader and get results
    i = 0
    for dat in val_loader:
        
        # Get inputs and target
        shortts = True if 'shortts' in file_names[0] else False
        if shortts:
            X, y, X_og, y_og, X_longts, y_longts = dat
            X, y, X_og, y_og, X_longts, y_longts = X.to(device), y.to(device), X_og.to(device), y_og.to(device), X_longts.to(device), y_longts.to(device)
        else:
            X, y = dat
            X, y = X.to(device), y.to(device)
            if torch.is_tensor(y):
                y = y.cpu().detach().numpy()

        # Pass normalized inputs to model, re-scale predictions
        X_norm = (X - torch.min(X[~torch.isnan(X)])) / (torch.max(X[~torch.isnan(X)]) - torch.min(X[~torch.isnan(X)]))
        preds_norm = model(X_norm).cpu().detach()
        preds = preds_norm * (torch.max(X[~torch.isnan(X)]) - torch.min(X[~torch.isnan(X)])) + torch.min(X[~torch.isnan(X)]) # re-scale 
        preds = preds.numpy()
        
        # Save preds
        if preds.shape[0] > 1: # if not pixel-wise
            for j in range(np.shape(preds)[0]):
                
                # Save inputs, predictions, and trues
                np.save(f'{save_dir}/x_{i}', X[j].cpu().detach().numpy())
                np.save(f'{save_dir}/true_{i}', np.array(y[j]))
                np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
                
                # If short-timescale-only model, save full and long-ts-only inputs and trues too
                if shortts:
                    np.save(f'{save_dir}/x_og_{i}', X_og[j].cpu().detach().numpy())
                    np.save(f'{save_dir}/true_og_{i}', np.array(y_og[j]))
                    np.save(f'{save_dir}/x_longts_{i}', X_longts[j].cpu().detach().numpy())
                    np.save(f'{save_dir}/true_longts_{i}', np.array(y_longts[j]))

                # Save start times
                if file_names != None:
                    starttimes_dict[i] = starttimes[i]
                i += 1  
        else:
            np.save(f'{save_dir}/x_{i}', X.cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y))
            np.save(f'{save_dir}/pred_{i}', np.array(preds))
            i += 1 
            
            
    # Save start times dict
    if file_names != None:
        pickle.dump(starttimes_dict, open(f'{save_dir}/starttimes.pkl','wb'))
    

    
def save_combined_model_results(val_loader, file_names, save_dir, model1, model2, r0, device='cpu', prnt=True):
    '''
    Run each validation obs through two models, combined (use one for center, one for outside), and save results
    For smooth stitching, use weighting function for pixels directly on either side of radius boundary
    NOTE: Expects un-normalized data in loader, so that we can save images with their original value ranges
    '''
    
    # Make dir
    if prnt: print(f'Saving results on validation data for combined model in {save_dir}')
    if os.path.exists(save_dir) == False: 
        os.makedirs(save_dir)
        
    # Load file names
    starttimes = [f[f.find('/')+1:f.find('_to')] for f in file_names] 
    starttimes_dict = {}
    
    # Loop over val loader and get results
    i = 0
    for X, y in val_loader:
        
        # Get inputs and target
        X, y = X.to(device), y.to(device)
        if torch.is_tensor(y):
            y = y.cpu().detach().numpy()
            
        # Normalized inputs to model
        X_min = float(torch.min(X))
        X_max = float(torch.max(X))
        X_norm = (X - X_min) / (X_max - X_min)
        
        # Get results of both models
        preds1_norm = model1(X_norm).cpu().detach().numpy() # for the outside
        preds2_norm = model2(X_norm).cpu().detach().numpy() # for the inside
        
        # Save preds
        for j in range(np.shape(preds1_norm)[0]): # loop over batch dim
            
            preds1 = preds1_norm[j]
            preds2 = preds2_norm[j]
            
            # Combine together
            n_pix = preds1.shape[-1]
            xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
            ctr_x, ctr_y = n_pix/2, n_pix/2
            r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
            preds = np.where(r < r0, preds2, preds1) 

            # For smooth stitching, "blur" around r0 by using a wieghted average
            preds = np.where((r0-2 < r) & (r < r0+2), ((r-r0+2)/4)*preds1 + (1-((r-r0+2)/4))*preds2, preds) # (r-r0+2)/4 -> 0 at r0-2 and 1 at r0+2

            # Re-scale predictions
            preds = preds * (X_max - X_min) + X_min # re-scale 
            
            # Save
            np.save(f'{save_dir}/x_{i}', X[j].cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds))
            starttimes_dict[i] = starttimes[i]
            i += 1    
            
            print(f'Saved {i}/{len(val_loader)*np.shape(preds1_norm)[0]}', end='\r')
            
    # Save start times dict
    print('Saving starttimes')
    pickle.dump(starttimes_dict, open(f'{save_dir}/starttimes.pkl','wb'))
    