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
    
    # Re-scale ctr_wght and otr_wghts if ctr_wght less than 1 [THIS SHOULD BE UNNECCSARY]
    otr_wght = 1
    if ctr_wght < 1: # if training to focus on outsides instead of centers
        otr_wght = 1/ctr_wght
        ctr_wght = 1

    # Get radius map to compute weights
    n_pix = pred.shape[2]
    xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    weights = torch.from_numpy(np.where(r < 30, ctr_wght, otr_wght))
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


# def CtrOnly_MSE(pred, truth, input):
#     '''
#     Compute MSE only over pixels where input is zero (i.e., missing data locations)
#     ONLY compute loss for central pixels
#     '''
    
#     # Get radius map to compute weights
#     n_pix = pred.shape[2]
#     xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
#     ctr_x, ctr_y = n_pix/2, n_pix/2
#     r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
#     weights = torch.from_numpy(np.where(r < 33, 1, 0))

#     # Compute loss
#     tot_loss = 0
#     for batch in range(input.shape[0]):

#         use_idxs = np.where(np.max(input[batch].numpy(), axis=(1,2)) == np.min(input[batch].numpy(), axis=(1,2)))[0] # Missing images won't still all be zero after norm, so just check if all values the same 

#         loss = 0
#         for i in use_idxs:
#             pred_use = pred[batch, i, :, :]
#             truth_use = truth[batch, i, :, :]
#             mse_pix = nn.MSELoss(reduction='none')(pred_use, truth_use)
#             mse = mse_pix * weights
#             loss += mse.mean()
        
#         tot_loss += loss/len(use_idxs)

#     return tot_loss/input.shape[0]


def CtrOnly_MSE(pred, truth, input):
    '''
    Compute MSE only over pixels where input is zero (i.e., missing data locations)
    ONLY compute loss for central pixels
    '''
    
    # Get radius map and get idxs within radius
    n_pix = pred.shape[2]
    xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    within_r_idxs = np.where(r < 33)
    within_r_x, within_r_y = within_r_idxs[0], within_r_idxs[1]

    # Compute loss
    tot_loss = 0
    for batch in range(input.shape[0]):

        use_idxs = np.where(np.max(input[batch].numpy(), axis=(1,2)) == np.min(input[batch].numpy(), axis=(1,2)))[0] # Missing images won't still all be zero after norm, so just check if all values the same 

        loss = 0
        for i in use_idxs:
            pred_use = pred[batch, i, within_r_x, within_r_y]
            truth_use = truth[batch, i, within_r_x, within_r_y]
            mse = nn.MSELoss()(pred_use, truth_use)
            loss += mse.mean()
        
        tot_loss += loss/len(use_idxs)

    return tot_loss/input.shape[0]


def OtrOnly_MSE(pred, truth, input):
    '''
    Compute MSE only over pixels where input is zero (i.e., missing data locations)
    ONLY compute loss for outer pixels
    '''
    
    # Get radius map to compute weights
    n_pix = pred.shape[2]
    xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    weights = torch.from_numpy(np.where(r > 27, 1, 0))

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