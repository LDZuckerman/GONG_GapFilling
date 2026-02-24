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

