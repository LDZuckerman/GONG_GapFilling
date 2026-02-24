import os
import pandas as pd
import json
import numpy as np

def get_modelDF(modeldir='', tag='', metric='RMSE'):
    '''
    Helper function to create dataframe of all run models, their parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f and '.ipynb_checkpoints' not in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        
        # Add val mse
        val_metric = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds', metric=metric)
        exp_dict[metric] = val_metric
    
        # Add to dict
        all_info.append(exp_dict)
    
    all_info = pd.DataFrame(all_info)
    
    return all_info 


def prediction_validation_results(output_dir, metric):
    '''
    Compute average error on validation predictions 
    '''

    truefiles = [file for file in os.listdir(output_dir) if 'true' in file]
    predfiles = [file for file in os.listdir(output_dir) if 'pred' in file]
    inputfiles = [file for file in os.listdir(output_dir) if 'x' in file]

    if metric == 'RMSE':

        tot_rmse = 0
        for i in range(len(truefiles)):
            true = np.load(f'{output_dir}/{truefiles[i]}').flatten()
            pred = np.load(f'{output_dir}/{predfiles[i]}').flatten()
            tot_rmse += np.sqrt(np.nanmean((true-pred)**2))
        out = tot_rmse/len(truefiles)

    elif metric == 'Selected_RMSE':
    
        for i in range(len(truefiles)):
            true = np.load(f'{output_dir}/{truefiles[i]}')
            pred = np.load(f'{output_dir}/{predfiles[i]}')
            input = np.load(f'{output_dir}/{inputfiles[i]}')

            #use_idxs = np.where(np.sum(input, axis=(1,2)) == 0)[0] # No batch dim, right?
            use_idxs = np.where(np.max(input, axis=(1,2)) == np.min(input, axis=(1,2)))[0] # WONT STILL BE ZEROS FTER NORMLIZATION, SO JUST CHECK IF ALL THE SAME

            loss = 0
            for i in use_idxs:
                pred_use = pred[i, :, :].flatten() # No batch dim, right?
                truth_use = true[i, :, :].flatten()
                loss += np.sqrt(np.nanmean((truth_use-pred_use)**2))
            out = loss/len(use_idxs)
        
    return out

def display_DF(DF, ignore_cols):
    '''
    Helper function to display all-model dataframe
    '''
    DF = DF.drop(columns=ignore_cols)
    
    display(DF)