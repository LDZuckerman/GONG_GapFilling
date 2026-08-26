import torch
import torch.nn as nn
import numpy as np

def MSE(pred, truth):
    '''
    Just return traditional MSE 
    '''
    mse = nn.MSELoss()

    return mse(pred, truth)


def Selected_MSE(pred, truth, input):
    '''
    Compute MSE only over pixels where input is zero (i.e., missing data locations)
    '''

    tot_loss = 0
    for batch in range(input.shape[0]):

        use_idxs = np.where(np.max(input[batch].numpy(), axis=(1,2)) == np.min(input[batch].numpy(), axis=(1,2)))[0] # Missing images won't still all be zero after norm, so just check if all values the same 

        loss = 0
        for i in use_idxs:
            pred_use = pred[batch, i, :, :]
            truth_use = truth[batch, i, :, :]
            loss += nn.MSELoss()(pred_use.flatten(), truth_use.flatten())
        
        tot_loss += loss/len(use_idxs)

    return tot_loss/input.shape[0]


def Regional_MSE(pred, truth, input, ctr_wght):
    '''
    Compute MSE only over pixels where input is zero (i.e., missing data locations)
    Allow for increased weighting of loss for low radius pixels
    '''
    
    # Get radius map to compute weights
    n_pix = pred.shape[2]
    xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    weights = torch.from_numpy(np.where(r < 30, ctr_wght, 1))
    # max_r =  61 # approx max radius (in pix) inside which pixels are sun, not background
    # sun_idxs = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
    # sun_mask = np.zeros_like(ex_pred) * np.NaN
    # sun_mask[sun_idxs] = 1 

    # Compute loss
    tot_loss = 0
    for batch in range(input.shape[0]):

        use_idxs = np.where(np.max(input[batch].numpy(), axis=(1,2)) == np.min(input[batch].numpy(), axis=(1,2)))[0] # Missing images won't still all be zero after norm, so just check if all values the same 

        loss = 0
        for i in use_idxs:
            pred_use = pred[batch, i, :, :]
            truth_use = truth[batch, i, :, :]
            mse_pix = nn.MSELoss(reduction='none')(pred_use, truth_use)
            mse = mse_pix * weights
            loss += mse.mean()
        
        tot_loss += loss/len(use_idxs)

    return tot_loss/input.shape[0]

