import os
import pandas as pd
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pickle as pkl
from astropy.io import fits
import torch
import sklearn
from sklearn.metrics import r2_score
import scipy.stats as stats
from torch.utils.data import Dataset, DataLoader
import sys 
sys.path.append('/pl/active/NSO-IT/data/leah/Solar_GapFilling/GONG_GapFilling/')
from utils import models, run_utils, data_utils


def get_modelDF(modeldir='', tag='', dtag=None, print_skipped=True, redo_metrics=False):
    '''
    Create dataframe of all run models, their parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f and '.ipynb_checkpoints' not in f and 'ignore' not in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Load exp dict and use if dataset = dtag
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        exp_dict['dataset'] = 'NN_Data_Ini1819_15_03_rand' if 'dataset' not in exp_dict else exp_dict['dataset']
        if exp_dict['dataset'] != 'NN_Data_'+dtag:
            continue
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds_scale'):
            if print_skipped:
                print(f'Skipping {expdir}; not finished training')
            continue
        
        # Add val metrics
        if not os.path.exists(f'{modeldir}/{expdir}/val_metrics.pkl') or redo_metrics:
            print(f'Computing validation set metrics for {modeldir}/{expdir}')
            if 'pixels' not in dtag:
                val_metrics = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds_scale') # metrics on normalized val preds
            else:
                val_metrics = prediction_validation_results_1D(output_dir=f'{modeldir}/{expdir}/test_preds_scale')
            pickle.dump(val_metrics, open(f'{modeldir}/{expdir}/val_metrics.pkl', 'wb'))
        else:
            val_metrics = pickle.load(open(f'{modeldir}/{expdir}/val_metrics.pkl', 'rb')) # metrics on normalized val preds
        for k in val_metrics.keys():
            exp_dict[k] = val_metrics[k]
        
        # Add parameters that were not tunable in earlier iterations
        exp_dict['dataset'] = exp_dict['dataset'].replace('NN_Data_','')
        exp_dict['kernel_size'] = 'NaN' if 'kernel_size' not in exp_dict else exp_dict['kernel_size']
        exp_dict['padding_mode'] = 'NaN' if 'padding_mode' not in exp_dict else exp_dict['padding_mode']
        exp_dict['img_size'] = 128 if 'img_size' not in exp_dict else exp_dict['img_size']
        exp_dict['batch_size'] = 16 if 'batch_size' not in exp_dict else exp_dict['batch_size']
        exp_dict['ctr_wgt'] = 'NaN' if 'ctr_wgt' not in exp_dict else exp_dict['ctr_wgt']
    
        all_info.append(exp_dict)
    
    # Create df and sort
    if len(all_info) > 0:
        all_info = pd.DataFrame(all_info).drop(columns=['img_size','batch_size','padding_mode','kernel_size'])
        all_info = all_info.sort_values(by=['name'])
    else:
        print(f'No models for given tag ({tag}) and dtag ({dtag})')
          
    # Add row for interp as comparison
    if 'pixels' not in dtag:
        if os.path.exists(f'{modeldir}/linear_interpolation/LI_metrics_{dtag}.pkl'):
            LI_metrics = pickle.load(open(f'{modeldir}/linear_interpolation/LI_metrics_{dtag}.pkl', 'rb'))
        else:
            print(f'Computing metrics for linear interp trained on dataset {dtag}')
            LI_metrics = get_li_metrics(dataset=f'NN_Data_{dtag}') # since I generally break the tables out by DS, 'dataset' from any exp_dict will do
            pickle.dump(LI_metrics, open(f'{modeldir}/linear_interpolation/LI_metrics_{dtag}.pkl', 'wb'))
        s = pd.DataFrame(columns=['name','dataset'], data=[['LI',f'{dtag}']])
        for k in LI_metrics.keys():
            s[k] = LI_metrics[k] 
        all_info = pd.concat([all_info, s])
    
    all_info = all_info.fillna('')
    
    return all_info 


def compare_across_dsets(mods, datasets, modeldir='', metric='rmse', redo_metrics=False):
    '''
    Create dataframe of the results of different models on different datasets
    '''
    
    all_dat = []
    idxs = []
    for moddir in mods:
        
        d = json.load(open(f'{modeldir}/{moddir}/exp_file.json','rb'))
        mod_dataset = d['dataset'] 
        idxs.append(f'{moddir} ({mod_dataset.replace("NN_Data_","").replace("_cluster","")})')
        
        dat = []
        for dataset in datasets:
            
            # Get metrics on this dataset
            dtag = '' if dataset == mod_dataset else f'_{dataset}'
            if not os.path.exists(f'{modeldir}/{moddir}/val_metrics{dtag}.pkl') or redo_metrics:
                
                print(f'Computing metrics for the performance of {moddir} on {dataset}')
                
                # Create val set preds if dont already exist
                dataset_preds_dir = f'{modeldir}/{moddir}/test_preds_scale{dtag}'
          
                if not os.path.exists(dataset_preds_dir):
                    print(f'   Getting predictions of {moddir} on {dataset}')           # freq_filter=d['freq_filter']
                    test_ds = data_utils.dataset(dataset=dataset, set='val', im_size=128, freq_filter=False, norm=False, dpath='../../Data') 
                    test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False) 
                    convblock_depth = 2 if 'convblock_depth' not in d.keys() else d['convblock_depth']
                    xs0, ys0 = next(iter(test_loader))
                    if d['model_name'] == 'UNet':
                        model = models.UNet(len_set=xs0.shape[1], convblock_depth=convblock_depth)
                    elif d['model_name'] == 'UNet2':
                        model = models.UNet2(len_set=xs0.shape[1], convblock_depth=convblock_depth)
                    model.load_state_dict(torch.load(f'{modeldir}/{moddir}/{moddir}.pth', weights_only=True, map_location=torch.device('cpu')))
                    run_utils.save_model_results(test_loader, save_dir=dataset_preds_dir , model=model, prnt=False)
                
                # Save metric
                val_metrics = prediction_validation_results(output_dir=dataset_preds_dir)
                pickle.dump(val_metrics, open(f'{modeldir}/{moddir}/val_metrics{dtag}.pkl','wb'))
                    
            else:   
                val_metrics = pickle.load(open(f'{modeldir}/{moddir}/val_metrics{dtag}.pkl','rb'))
                
            # Append desired metric
            m = val_metrics[metric]
            dat.append(m)
        
        all_dat.append(dat)
        
    # Make DF
    df = pd.DataFrame(all_dat, columns=datasets, index=idxs)
    
    return df


def get_LIDF(datasets=['2008_40_20_cluster','2019_40_20_cluster','2024_40_20_cluster','2019_20_10_cluster','2024_20_10_cluster','2019_15_03_cluster', '2024_15_03_cluster']):
    '''
    Helper function to create a dataframe of the results of performing simple linear interp on all datasets
    '''
    
    all_info = []
    for ds in datasets:
        
        # If LI metrics already exist for this dataset, use them
        if os.path.exists(f'../../model_runs/linear_interpolation/LI_metrics_{ds}.pkl'):
            LI_metrics = pickle.load(open(f'../../model_runs/linear_interpolation/LI_metrics_{ds}.pkl', 'rb'))
            
        # If not, compute LI predictions for this dataset and then compute metrics
        else:
            print(f'Computing metrics for linear interp trained on dataset NN_Data_{ds}')
            LI_metrics = get_li_metrics(dataset=f'NN_Data_{ds}') # since I generally break the tables out by DS, 'dataset' from any exp_dict will do
            pickle.dump(LI_metrics, open(f'../../model_runs/linear_interpolation/LI_metrics_{ds}.pkl', 'wb'))  
        
        # Make dict and append
        d = {'dataset':f'NN_Data_{ds}'} | LI_metrics
        all_info.append(d)
        
    # Make DF
    all_info = pd.DataFrame(all_info)
    
    return all_info
    

def prediction_validation_results(output_dir):
    '''
    Compute average error metrics on validation predictions
    NOTE: normalize first so that metrics are on normalized data 
    '''

    # Get all true, pred, and input files
    truefiles = [file for file in np.sort(os.listdir(output_dir)) if 'true' in file]
    predfiles = [file for file in np.sort(os.listdir(output_dir)) if 'pred' in file]
    inputfiles = [file for file in np.sort(os.listdir(output_dir)) if 'x' in file]
    
    # Initialize metrics and dict
    val_metrics = {}
    tot_selected_rmse = 0
    tot_selected_r2 = 0
    tot_selected_rho = 0
    tot_selected_mae = 0
    
    # Get radius map to exclude background pix in images
    ex_pred = np.load(f'{output_dir}/{predfiles[0]}')
    length = ex_pred.shape[0]
    n_pix = ex_pred.shape[1]
    zz, xx, yy = np.meshgrid(np.linspace(0, length-1, length), np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    max_r =  61 # approx max radius (in pix) inside which pixels are sun, not background
    sun_idxs = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
    sun_mask = np.zeros_like(ex_pred) * np.NaN
    sun_mask[sun_idxs] = 1 

    # Loop over sets
    for i in range(len(truefiles)):

        # Load true, pred, and input set
        true = np.load(f'{output_dir}/{truefiles[i]}')
        pred = np.load(f'{output_dir}/{predfiles[i]}')
        inp = np.load(f'{output_dir}/{inputfiles[i]}')
        
        # Normalize
        try:
            minimum = torch.min(true); maximum =torch.max(true)
        except TypeError:
            minimum = np.min(true); maximum = np.max(true)
        true = (true - minimum) / (maximum - minimum)
        pred = (pred - minimum) / (maximum - minimum)
        
        # Remove (set to NaN) pixels in background
        pred = pred * sun_mask
        true = true * sun_mask

        # Get idxs of the gaps
        use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] # WONT STILL BE ZEROS AFTER NORMLIZATION, SO JUST CHECK IF ALL THE SAME

        # Compute metrics
        smrse = 0
        sr2 = 0
        srho = 0
        smae = 0
        for j in use_idxs:
            
            # Flatten and drop NaNs (for r2 score)
            use_pred = pred[j, :, :].flatten() 
            use_true = true[j, :, :].flatten()
            use_pred = use_pred[~np.isnan(use_pred)]
            use_true = use_true[~np.isnan(use_true)]
            
            # Compute metrics
            smrse += np.sqrt(np.mean((use_true-use_pred)**2))
            sr2 += sklearn.metrics.r2_score(use_true, use_pred)
            srho += stats.spearmanr(use_true, use_pred)[0] # stats.pearsonr(use_true, use_pred.flatten)[0]
            smae += np.mean(np.abs(use_true-use_pred))

        tot_selected_rmse += smrse/len(use_idxs)
        tot_selected_r2 += sr2/len(use_idxs)
        tot_selected_rho += srho/len(use_idxs)
        tot_selected_mae += smae/len(use_idxs)
        
    out = {'rmse':tot_selected_rmse/len(truefiles), 'mae':tot_selected_mae/len(truefiles), 'r2':tot_selected_r2/len(truefiles), 'rho':tot_selected_rho/len(truefiles)}
    
    return out


def prediction_validation_results_1D(output_dir):
    '''
    Compute average error metrics on validation predictions
    NOTE: normalize first so that metrics are on normalized data 
    '''

    # Get all true, pred, and input files
    truefiles = [file for file in np.sort(os.listdir(output_dir)) if 'true' in file]
    predfiles = [file for file in np.sort(os.listdir(output_dir)) if 'pred' in file]
    inputfiles = [file for file in np.sort(os.listdir(output_dir)) if 'x' in file]
    
    # Initialize metrics and dict
    val_metrics = {}
    tot_selected_rmse = 0
    tot_selected_r2 = 0
    tot_selected_rho = 0
    tot_selected_mae = 0

    # Loop over sets
    for i in range(len(truefiles)):

        # Load true, pred, and input set
        true = np.squeeze(np.load(f'{output_dir}/{truefiles[i]}'))
        pred = np.squeeze(np.load(f'{output_dir}/{predfiles[i]}'))
        inp = np.squeeze(np.load(f'{output_dir}/{inputfiles[i]}'))
        #print(inp.shape, true.shape, pred.shape)
        
        # Normalize
        try:
            minimum = torch.min(true); maximum =torch.max(true)
        except TypeError:
            minimum = np.min(true); maximum = np.max(true)
        true = (true - minimum) / (maximum - minimum)
        pred = (pred - minimum) / (maximum - minimum)

        # Get idxs of the gaps
        use_idxs = np.where(np.isnan(inp))[0] # WONT STILL BE ZEROS AFTER NORMLIZATION, SO JUST CHECK IF ALL THE SAME
        
        # Get true and pred at those idxs
        use_true = true[use_idxs]
        use_pred = pred[use_idxs]

        # Compute metrics
        smrse = np.sqrt(np.mean((use_true-use_pred)**2))
        sr2 = sklearn.metrics.r2_score(use_true, use_pred)
        srho = stats.spearmanr(use_true, use_pred)[0] # stats.pearsonr(use_true, use_pred.flatten)[0]
        smae = np.mean(np.abs(use_true-use_pred))

        tot_selected_rmse += smrse/len(use_idxs)
        tot_selected_r2 += sr2/len(use_idxs)
        tot_selected_rho += srho/len(use_idxs)
        tot_selected_mae += smae/len(use_idxs)
        
    out = {'rmse':tot_selected_rmse/len(truefiles), 'mae':tot_selected_mae/len(truefiles), 'r2':tot_selected_r2/len(truefiles), 'rho':tot_selected_rho/len(truefiles)}
    
    return out


def display_DF(DF, ignore_cols):
    '''
    Helper function to display all-model dataframe
    '''
    DF = DF.drop(columns=ignore_cols)
    
    display(DF)
    
        
def predict_validation_day(inputs, model_name, plot_check=None, trues=None):
    '''
    Fill all the gaps in a given day (1440 long sequence)
    '''
    
    # If single model, load model
    if 'combined' not in model_name:
        
        # Load model
        exp_outdir = f'../../model_runs/{model_name}'
        d = json.load(open(f'{exp_outdir}/exp_file.json', 'rb'))
        len_set = int(d['dataset'].split('_')[3]) # int(json.load(open(f'../../model_runs/{model_name}/exp_file.json'))['dataset'][13:15])
        convblock_depth = 2 if 'convblock_depth' not in d.keys() else d['convblock_depth']
        if d['model_name'] == 'UNet':
            model = models.UNet(len_set=len_set, convblock_depth=convblock_depth)
        elif d['model_name'] == 'UNet2':
            model = models.UNet2(len_set=len_set, convblock_depth=convblock_depth)
        mod_pth = f'{exp_outdir}/{model_name}.pth' # f'{exp_outdir}/{model_name}.pth'
        model.load_state_dict(torch.load(mod_pth, map_location=torch.device('cpu'), weights_only=False))
    
    # If combining, load models and set switch radius
    else:
        
        # Load models
        mod1 = model_name[0:6]  # UNet14UNet27combined_r30
        mod2 = model_name[6:12]
        exp1_outdir = f'../../model_runs/{mod1}/'
        d1 = json.load(open(f'{exp1_outdir}/exp_file.json','rb'))
        len_set = int(d1['dataset'].split('_')[3])
        convblock_depth = 2 if 'convblock_depth' not in d1.keys() else d1['convblock_depth']
        if d1['model_name'] == 'UNet':
            model1 = models.UNet(len_set=len_set, convblock_depth=convblock_depth)
        elif d1['model_name'] == 'UNet2':
            model1 = models.UNet2(len_set=len_set, convblock_depth=convblock_depth)
        model1.load_state_dict(torch.load(f'{exp1_outdir}/{d1["name"]}.pth', map_location=torch.device('cpu'), weights_only=True))
        exp2_outdir = f'../../model_runs/{mod2}/'
        d2 = json.load(open(f'{exp2_outdir}/exp_file.json','rb'))
        convblock_depth = 2 if 'convblock_depth' not in d2.keys() else d2['convblock_depth']
        if d2['model_name'] == 'UNet':
            model2 = models.UNet(len_set=len_set, convblock_depth=convblock_depth)
        elif d2['model_name'] == 'UNet2':
            model2 = models.UNet2(len_set=len_set, convblock_depth=convblock_depth)
        model2.load_state_dict(torch.load(f'{exp2_outdir}/{d2["name"]}.pth', map_location=torch.device('cpu'), weights_only=True))
        
        # Compute radii
        r0 = int(model_name[-2:])
        n_pix = 209
        xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
        ctr_x, ctr_y = n_pix/2, n_pix/2
        r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))

    # Find the locations of gaps in the input, create sets around those gaps, fill with model
    preds = np.copy(inputs) # Could alternatively copy trues here, but they are the same at non-gap idxs
    gap_idxs = np.where(np.sum(inputs, axis=(1,2)) == 0)[0]
    diffs = np.diff(gap_idxs) # gives gap_idxs[i]-gap_idxs[i-1] for all i>0 (so length is len(gap_idxs)-1)
    split_idxs = [i+1 for i in range(len(diffs)) if diffs[i]> 1]
    gap_sections = np.array_split(gap_idxs, split_idxs) # array of arrays, with each array being idxs of given gap
    gap_ctrs = [int(np.mean(gap_idxs)) for gap_idxs in gap_sections]
    
    # Print warning if there are gap sections over the stated max length
    n_too_long = sum([len(gap_idxs) > len_set/2 for gap_idxs in gap_sections])
    if n_too_long > 0:
        print(f'      Warning: this day contains {n_too_long} gaps that are longer than the {int(len_set/2)} the model has been trained on')

    # Loop over the gap ctrs
    for i in range(len(gap_ctrs)):

        # Get the input section at the len_set idxs centered on the center of the gap
        gap_idxs = gap_sections[i]
        gap_ctr = gap_ctrs[i]
        n = int(len_set/2)
        sec_idxs = list(np.linspace(gap_ctr-n+1, gap_ctr-1, n-1, dtype=int)) + [gap_ctr] + list(np.linspace(gap_ctr+1, gap_ctr+n, n, dtype=int)) # model input section
        sec_start = sec_idxs[0]
        sec_end = sec_idxs[1]
        section = inputs[sec_idxs,:,:]

        # Deal with the case where there are gaps in the non-gap portions of the input
        # In the mock data, this would happen if the gaps are too close together
        # For now, just use LI to remove unexpected gaps. In the future, train models to work on gaps at any position. 
        expected_nongap_idxs = [idx for idx in sec_idxs if idx not in gap_idxs]
        sec_expected_non_gap = section[expected_nongap_idxs-sec_start] # np.delete(section, expected_gap_idxs, axis=0) # section[~expected_gap_idxs,:,:]
        expected_filled_have_gaps = np.sum(sec_expected_non_gap, axis=(1, 2)) == 0
        if np.any(expected_filled_have_gaps):  
            starts = np.where(expected_filled_have_gaps, np.maximum.accumulate(np.where(expected_filled_have_gaps & ~np.concatenate(([False], expected_filled_have_gaps[:-1])), expected_nongap_idxs, -1)), np.nan)
            ends = np.where(expected_filled_have_gaps, np.minimum.accumulate(np.where(expected_filled_have_gaps & ~np.r_[expected_filled_have_gaps[1:], False], expected_nongap_idxs, np.inf)[::-1])[::-1], np.nan)
            starts = np.unique(starts[~np.isnan(starts)]) #+ sec_start
            ends = np.unique(ends[~np.isnan(ends)]) #+ sec_start
            for j in range(len(starts)):
                true_start_idx = starts[j] # idx in the ORIGINAL 1440 seqence 
                true_end_idx = ends[j] # idx in the ORIGINAL 1440 seqence 
                unexp_gap_idxs = np.linspace(true_start_idx, true_end_idx, int(true_end_idx-true_start_idx)+1)
                # Find before-gap image (if section starts with gap, must check backwards through preceeding images)
                if true_start_idx != sec_idxs[0]:
                    before_gap_img = inputs[int(true_start_idx-1),:,:]
                    idx_in_gap = 0 # the first idx of the gap we see is indeed the first idx of the gap, so start at 0
                else:
                    before_gap_img = np.zeros((209, 209))
                    count = 1
                    while np.sum(before_gap_img) == 0:
                        before_gap_img = inputs[int(true_start_idx-1-count),:,:]
                        count += 1
                    idx_in_gap = count # the first idx of the gap we see is NOT the first idx of the gap, so start at how far it actually is into the gap
                # Find after-gap image (if section ends with gap, must check backwards through preceeding images)
                if true_end_idx != sec_idxs[-1]:
                    after_gap_img = inputs[int(true_end_idx+1),:,:]
                else:
                    after_gap_img = np.zeros((209, 209))
                    count = 1
                    while np.sum(after_gap_img) == 0:
                        after_gap_img = inputs[int(true_end_idx+1+count),:,:]
                        count += 1
                # Fill unexpected gap with LI
                filled_unexp_gap = np.empty((len(unexp_gap_idxs), 209, 209))
                for k in range(len(unexp_gap_idxs)):
                    weight = idx_in_gap/len(unexp_gap_idxs)
                    interp_img = (1 - weight) * before_gap_img + weight * after_gap_img
                    filled_unexp_gap[k] = interp_img
                    idx_in_gap += 1 
                section[unexp_gap_idxs.astype(int)-sec_start] = filled_unexp_gap

        # Normalize
        x = (section - np.min(section)) / (np.max(section) - np.min(section)) # normalize 

        # Get model predictions on that section 
        if 'combined' not in model_name:
            y = model(torch.tensor(np.expand_dims(x, axis=0)))
            y = np.squeeze(y.detach().numpy())
        else:
            y1 = model2(torch.tensor(np.expand_dims(x, axis=0)))
            y1 = np.squeeze(y1.detach().numpy())
            y2 = model2(torch.tensor(np.expand_dims(x, axis=0)))
            y2 = np.squeeze(y2.detach().numpy())
            y = np.where(r < r0, y2, y1) 
            y = np.where((r0-2 < r) & (r < r0+2), ((r-r0+2)/4)*y1 + (1-((r-r0+2)/4))*y2, y)
            
        # Put the model predictions in for the gap idxs in preds
        sec_pred = y * (np.max(section) - np.min(section)) + np.min(section) # re-scale 
        sec_gap_idxs = np.where(np.sum(section, axis=(1, 2)) == 0)[0] # idxs within the input section that are the gap
        preds[gap_idxs,:,:] = sec_pred[sec_gap_idxs]

        # Plot to check quailty 
        fig = None
        if plot_check:
            sec_mod_trues = trues[sec_idxs,:,:]
            fig, axs = plt.subplots(2, len_set, figsize=(len_set, 2))
            for i in range(len_set):
                im0 = axs[0,i].imshow(sec_mod_trues[i,:,:], cmap='gray'); axs[0,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
                if i in sec_gap_idxs:
                    axs[0,i].plot(np.linspace(0, 209, 209), np.linspace(0, 209, 209), c='red')
                im1 = axs[1,i].imshow(preds[sec_idxs,:,:][i,:,:], cmap='gray'); axs[1,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 

    return preds, fig



def get_li_metrics(dataset):
    '''
    Compute average error metrics on validation predictions
    '''
    
    # Get all true, pred, and input files
    output_dir = f'../../model_runs/linear_interpolation/LI_test_preds_{dataset}_new/'
    if not os.path.exists(output_dir):
        save_li_filled_set(dataset)
        
    print('   Computing metrics')
    truefiles = [file for file in np.sort(os.listdir(output_dir)) if 'true' in file]
    predfiles = [file for file in np.sort(os.listdir(output_dir)) if 'pred' in file]
    inputfiles = [file for file in np.sort(os.listdir(output_dir)) if 'x' in file]
    
    # Initialize metrics and dict
    val_metrics = {}
    tot_selected_rmse = 0
    tot_selected_r2 = 0
    tot_selected_rho = 0
    
    # Loop over sets
    for i in range(len(truefiles)):
        
        print(f'      {i}/{len(truefiles)}', end='\r')

        # Load true, pred, and input set
        true = np.load(f'{output_dir}/{truefiles[i]}')
        pred = np.load(f'{output_dir}/{predfiles[i]}')
        inp = np.load(f'{output_dir}/{inputfiles[i]}')

        # Get idxs of the gaps
        use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] # WONT STILL BE ZEROS AFTER NORMLIZATION, SO JUST CHECK IF ALL THE SAME

        # Compute metrics
        smrse = 0
        sr2 = 0
        srho = 0
        for i in use_idxs:
            use_pred = pred[i, :, :].flatten() # No batch dim, right?
            use_true = true[i, :, :].flatten()
            smrse += np.sqrt(np.nanmean((use_true-use_pred)**2))
            sr2 += r2_score(use_true, use_pred)
            srho += stats.spearmanr(use_true, use_pred)[0] # stats.pearsonr(use_true, use_pred.flatten)[0]

        tot_selected_rmse += smrse/len(use_idxs)
        tot_selected_r2 += sr2/len(use_idxs)
        tot_selected_rho += srho/len(use_idxs)
        
    out = {'rmse':tot_selected_rmse/len(truefiles), 'r2':tot_selected_r2/len(truefiles), 'rho':tot_selected_rho/len(truefiles)}
    
    return out           


            
def save_li_filled_set(dataset):
    '''
    Fill gapped dataset using simple linear interpolation (mean of boundaries)
    '''
    
    print(f'   Saving LI validation predictions for dataset {dataset}')
    
    # Get all true, pred, and input files
    ex_model_dir = get_mod_dir_with_given_ds(dataset) # dir to get trues and inputs from, e.g. a model with the same dataset
    inputfiles = [file for file in np.sort(os.listdir(f'../../model_runs/{ex_model_dir}/test_preds_scale/')) if 'x' in file]
    truefiles = [file for file in np.sort(os.listdir(f'../../model_runs/{ex_model_dir}/test_preds_scale/')) if 'true' in file] # just for saving for convenience
    li_pred_outdir = f'../../model_runs/linear_interpolation/LI_test_preds_{dataset}_new/'
    if not os.path.exists(li_pred_outdir):
        os.mkdir(li_pred_outdir)
    
    # Loop over sets
    for i in range(len(inputfiles)):
        
        print(f'      {i}/{len(truefiles)}', end='\r')

        inp = np.load(f'../../model_runs/{ex_model_dir}/test_preds_scale/{inputfiles[i]}')
        tru = np.load(f'../../model_runs/{ex_model_dir}/test_preds_scale/{truefiles[i]}') # just for saving for convenience
        li_pred = np.copy(tru)
        
        gap_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0]
        before_gap_img = inp[gap_idxs[0]-1]
        after_gap_img = inp[gap_idxs[-1]+1]
        idx_in_gap = 0
        for j in gap_idxs:
            weight = idx_in_gap/len(gap_idxs)
            interp_img = (1 - weight) * before_gap_img + weight * after_gap_img
            li_pred[j] = interp_img
            idx_in_gap += 1
            
        np.save(f'{li_pred_outdir}/{inputfiles[i].replace("x","pred")}', li_pred)
        np.save(f'{li_pred_outdir}/{inputfiles[i]}', inp) # saving for convenience
        np.save(f'{li_pred_outdir}/{truefiles[i]}', tru) # saving for convenience
        

        
def predict_validation_day_LI(inputs, save_path):
    '''
    Fill all the gaps in a given day (1440 long sequence) WITH LINEAR INTERPOLATION
    '''
    
    # Check if already exists
    if os.path.exists(save_path):
        return pickle.load(open(save_path, "rb"))
    
    # Find the locations of gaps in the input, create sets around those gaps, fill with model
    li_preds = np.copy(inputs) # np.empty((1440, 209, 209))
    gap_idxs = np.where(np.sum(inputs, axis=(1,2)) == 0)[0]
    diffs = np.diff(gap_idxs) # gives gap_idxs[i]-gap_idxs[i-1] for all i>0 (so length is len(gap_idxs)-1)
    split_idxs = [i+1 for i in range(len(diffs)) if diffs[i]> 1]
    gap_sections = np.array_split(gap_idxs, split_idxs) # array of arrays, with each array being idxs of given gap
    #gap_ctrs = [int(np.mean(gap_idxs)) for gap_idxs in gap_sections]
    
    # Loop over the gap ctrs
    for i in range(len(gap_sections)):

        # Get LI on that section and put the LI in for the gap idxs in preds
        gap_idxs = gap_sections[i]
        li_pred = np.copy(inputs[gap_idxs])
        before_gap_img = inputs[gap_idxs[0]-1] 
        after_gap_img = inputs[gap_idxs[-1]+1] 
        idx_in_gap = 0
        for j in range(len(gap_idxs)):
            weight = idx_in_gap/len(gap_idxs)
            interp_img = (1 - weight) * before_gap_img + weight * after_gap_img
            li_pred[j] = interp_img
            idx_in_gap += 1
        #sec_gap_idxs = np.where(np.sum(section, axis=(1, 2)) == 0)[0] # idxs within the input section that are the gap
        li_preds[gap_idxs,:,:] = li_pred
        
    # Save
    print(f'Saving predictions to {save_path}')
    pickle.dump(li_preds, open(save_path, 'wb'))

    return li_preds


def get_mod_dir_with_given_ds(dataset):
    
    all_mod_dirs = [f for f in os.listdir('../../model_runs') if '.ipynb' not in f and 'linear' not in f and '.pkl' not in f]
    
    found = False
    i = 0
    while not found and i < len(all_mod_dirs):
        mod_dir = all_mod_dirs[i]
        exp_dict = json.load(open(f'../../model_runs/{mod_dir}/exp_file.json','rb'))
        
        if mod_dir == 'UNet21': print(exp_dict['dataset'], dataset)
        
        if 'dataset' in exp_dict.keys() and exp_dict['dataset'] == dataset: 
            if 'test_preds_scale' in os.listdir(f'../../model_runs/{mod_dir}'):
                found = True
        i += 1
        
    if not found:
        raise ValueError(f'No models with dataset {dataset} (or those exp_dicts dont have the dataset key)')
          
    return mod_dir
     