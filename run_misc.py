#import pandas as pd
import numpy as np
import pickle
import json
import astropy.io.fits as fits
import os, shutil
import torch 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from itertools import islice
import argparse
import scipy.ndimage as sndi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
from utils import eval_utils, data_utils, run_utils, models

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True) 
parser.add_argument("-s", "--subset_folder", type=str, required=False)
parser.add_argument("-l", "--set_length", type=str, required=False)
parser.add_argument("-n", "--num_missing", type=str, required=False)
parser.add_argument("-m", "--sample_method", type=str, required=False)
parser.add_argument("-x", "--shortts", type=str, required=False)
parser.add_argument("-r", "--redo", type=str, default='', required=False)
parser.add_argument("-d", "--debug_mock", type=str, default='False', required=False)
args = parser.parse_args()

print(f'Performing task {args.task}')

if args.task == 'save_missing_img_names':
    
    data_subset=args.subset_folder #'Subset_2019'
    data_utils.save_missing_filenames(data_subset=data_subset, dpath='../Data')
    
    
elif args.task == 'create_dataset':
    
    print(f'Running dataset creation with the following arguements: {args}')
    
    from_folder = args.subset_folder # e.g. 'Subset_2024' or 'Subset_2018$Subset_2019$Subset_2020' or 'Subset_2019_shortts'
    set_length = int(args.set_length)
    num_missing = int(args.num_missing)
    sample_method = args.sample_method
    shortts = eval(args.shortts)
    redo = eval(args.redo)
    #debug_mock = eval(str(args.debug_mock))
    
    #from_tag = from_folder[-12:-8] if "shortts" in from_folder else from_folder[-4:] if "Subset" in from_folder else from_folder # e.g. '2019' from_folder[from_folder.find('Data')+5:from_folder.find(str(set_length))-1]
    from_tag = from_folder[-4:] if "Subset" in from_folder else from_folder
    print(f'Creating NN dataset from {from_folder}, using set_length = {set_length}, num_missing = {num_missing}, sample_method = {sample_method}, shortts = {shortts}')

    data_utils.create_dataset(from_folder, from_tag, set_length, num_missing, sample_method, redo, shortts=shortts, dpath='../Data')   

    
elif args.task == 'create_pixel_dataset':
    
    # Define dataset to take image sets from and parameters for selecting a subset of pixel signals from each set
    from_NN_set = 'NN_Data_2019_40_20_cluster'
    max_r = 50 # radius outside of which to ignore pixels (must be less than 61, approx r of backround start)
    n_per = 10 # pixels to select per set

    # Get all the true sets
    all_trues = np.sort([f for f in os.listdir(f'../Data/{from_NN_set}/') if '_true' in f])
    set_len = np.load(f'../Data/{from_NN_set}/{all_trues[0]}').shape[0]

    # Loop over the true sets, select some pixels from each set, and add the signals from those pixels to dict
    all_true_seqs = []
    all_inp_seqs = []
    count = 0
    for file in all_trues:
        
        # Get the true set from this file and the input set from the corresponding input file; find the gap idxs
        print(f'Processing file {count}/{len(all_trues)}')
        true_set = np.load(f'../Data/{from_NN_set}/{file}')
        inp_set = np.load(f'../Data/{from_NN_set}/{file.replace("_true", "_x")}')
        set_gap_idxs = np.where(np.max(inp_set, axis=(1,2)) == np.min(inp_set, axis=(1,2)))[0]  
        
        # Compute the pixels within the radius to use and select n_per pixels from those 
        zz, xx, yy = np.meshgrid(np.linspace(0, set_len-1, set_len), np.linspace(0, true_set.shape[1]-1, true_set.shape[1]), np.linspace(0, true_set.shape[1]-1, true_set.shape[1]), indexing='ij')
        ctr_x, ctr_y = true_set.shape[1]/2, true_set.shape[1]/2
        r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
        pix_in_r = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
        use_pix_idxs = np.random.choice(np.linspace(0, len(pix_in_r[0])-1, len(pix_in_r[0]), dtype=int), n_per)
        
        # For each of the selected pixels, add the signal over the set to dict
        for idx in use_pix_idxs:
            true_sequence = true_set[:, pix_in_r[1][idx], pix_in_r[2][idx]]
            inp_sequence = inp_set[:, pix_in_r[1][idx], pix_in_r[2][idx]]
            inp_sequence[set_gap_idxs] = np.NaN
            if np.all(inp_sequence==0):
                raise ValueError(f'Entire input appears to be 0 for pixel ({pix_in_r[0][idx]}, {pix_in_r[1][idx]}) of image {file}')
            all_true_seqs.append(true_sequence)
            all_inp_seqs.append(inp_sequence)
        count += 1
    dat = {'inputs':all_inp_seqs, 'trues':all_true_seqs}
    pickle.dump(dat, open(f'../Data/{from_NN_set}_pixels.pkl','wb'))
    
    # Save test idxs
    n_test = int(len(inputs)*0.2)
    test_idxs = np.random.choice(np.linspace(0,len(inputs)-1,len(inputs),dtype=int), n_test, replace=False)
    np.save(f'../Data/{from_NN_set}_pixels_test_idxs.npy', test_idxs)

    
elif args.task == 'zip_files':

    import shutil
    import logging

    zipfile_name = "2020" # "filled_10_preds_UNet18"
    folder_to_zip = "../Data/Originals/Subset_2020" # "Data/Full_Day_Val_Sets/filled_10_preds_UNet18"
    
    logging.basicConfig(level=logging.INFO)
    my_logger = logging.getLogger("archive_logger")

    compressed_file_path = shutil.make_archive(
        base_name=zipfile_name,
        format='gztar', # Specifies gzip compressed tar format
        root_dir=os.path.dirname(folder_to_zip), # Directory to start archiving from
        base_dir=os.path.basename(folder_to_zip), # Directory to be added to the archive
        logger=my_logger
    )
    
    
elif args.task == 'redo_vals':
    '''
    Redo just the validation results for all models
        - Had saved only the normalized values, and need the actual ranges to display rescaled 
        - Since need to loop through to get ranges anyway, probabaly best to just re-compute the validation results 
        - Then for future runs I have changed the things so that the validation performed during the model runs will save un-normalized data
    '''

    mod_names = np.sort([f for f in os.listdir('../model_runs') if 'linear' not in f and 'UNet' in f])
    for mod_name in mod_names:
        
        if not os.path.exists(f'../model_runs/{mod_name}/test_preds_scale'):
            
            print(f'Skipping {mod_name} - no existing preds')
            
            continue
            
        else:
            
            print(f'Re-doing validation preds for {mod_name}')

            d = json.load(open(f'../model_runs/{mod_name}/exp_file.json'))
            dataset = 'NN_Data_Ini1819_15_03_rand' if 'dataset' not in d else d['dataset']
            test_ds = data_utils.dataset(dataset=dataset, set='val', norm=False, im_size=128, freq_filter=False) # DONT NORMALIZE VAL SET
            test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False) # DONT SHUFFLE VAL SET
            xs0, _ = next(iter(test_loader))

            convblock_depth = 2 if 'convblock_depth' not in d.keys() else d['convblock_depth']
            model = models.UNet(len_set=xs0.shape[1], convblock_depth=convblock_depth)
            model.load_state_dict(torch.load(f'../model_runs/{mod_name}/{mod_name}.pth', map_location=torch.device('cpu')))
            model.eval()

            run_utils.save_model_results(test_loader, file_names=test_ds.x_sets, save_dir=f'../model_runs/{mod_name}/test_preds_new' , model=model) 
            #os.rename(f'../model_runs/{mod_name}/test_preds_old', f'../model_runs/{mod_name}/test_preds_norm') # rename old test_preds
            shutil.rmtree(f'../model_runs/{mod_name}/test_preds_scale') # remove old test_preds
            os.rename(f'../model_runs/{mod_name}/test_preds_new', f'../model_runs/{mod_name}/test_preds_scale') # rename new test_preds
    
    
elif args.task == 'predict_validation_days_LI':
    
    synthetic_set = 'filled_20'
    use_days = ['mrfqi090804', 'mrfqi090805','mrfqi091202','mrfqi211211','mrfqi220608','mrfqi260226']
    
    for day in np.sort(use_days):
        
        all_true_files = np.sort(os.listdir(f'../Data/Full_Day_Val_Sets/True_Full_Days/{day}'))
        day_inp = []
        for file in all_true_files:
            header = fits.open(f'../Data/Full_Day_Val_Sets/{synthetic_set}/{day}/{day}/{file}')[0].header
            true = fits.open(f'../Data/Full_Day_Val_Sets/True_Full_Days/{day}/{file}')[0].data
            day_inp.append(np.zeros((209, 209)) if 'FILLED' in header.keys() else true)
            
        if not os.path.exists(f'../Data/Full_Day_Val_Sets/{synthetic_set}_preds_LI/'):
            os.mkdir(f'../Data/Full_Day_Val_Sets/{synthetic_set}_preds_LI/')
            
        li_pred = eval_utils.predict_validation_day_LI(np.array(day_inp), save_path=f'../Data/Full_Day_Val_Sets/{synthetic_set}_preds_LI/{day}.pkl')
        
        
elif args.task == 'Save_Short_Timescale_Subsets':
    
    # Get all day folders for the year
    year = '2019'
    year_path = f'../Data/Originals/Subset_{year}'
    day_folders = np.sort([d for d in os.listdir(year_path) if 'mrfqi' in d])

    # Create new subset dir for the processed year
    if not os.path.exists(f'{year_path}_shortts'):
        os.mkdir(f'{year_path}_shortts')
    if not os.path.exists(f'{year_path}_longts'):
        os.mkdir(f'{year_path}_longts')

    # Process each day
    k = 0
    for day_folder in day_folders:
        
        print(f'  Processing day folder {k}/{len(day_folders)}'); k+=1

        ##################
        # Get the full day
        ##################

        min_files = np.sort([f for f in os.listdir(f'{year_path}/{day_folder}/')])
        full_day = np.empty((1440, 209, 209)) 
        for i in range(len(min_files)): # 1440
            min_file = min_files[i]
            full_day[i,:,:] = fits.open(f'{year_path}/{day_folder}/{min_file}')[0].data


        #########
        # Procces
        #########

        # Since even these "100%" DC sets aren't truely 100%, use LI to fill 
        gap_idxs = np.where(np.sum(full_day, axis=(1,2))==0)[0]
        valid_idxs = np.where(np.sum(full_day, axis=(1,2))!=0)[0]
        gong_seq_filled_linear = full_day.copy()
        lin_interp = interp1d(valid_idxs, full_day[valid_idxs,:,:] , axis=0, kind='linear',fill_value="extrapolate") # Add fill_value="extrapolate" so that it can predict gap idxs outside of valid idxs (e.g. if there is a missing img at the start or end of the day)
        interpolated_slice = lin_interp(gap_idxs)
        gong_seq_filled_linear[gap_idxs,:,:] = interpolated_slice

        # Get smoothed 
        gong_seq_filled_smooth = gong_seq_filled_linear.copy()
        gong_seq_smooth_gauss = ndimage.gaussian_filter1d(gong_seq_filled_linear, sigma=9, axis=0, mode='reflect')

        # Get smoothed removed
        gong_seq_trend_removed = full_day - gong_seq_smooth_gauss

        # Put gaps back in 
        gong_seq_trend_removed[gap_idxs,:,:] = 0.0

        ###########################################
        # Save back in original format as day files
        ###########################################

        if not os.path.exists(f'{year_path}_shortts/{day_folder}'):
            os.mkdir(f'{year_path}_shortts/{day_folder}')
            os.mkdir(f'{year_path}_longts/{day_folder}')
            
        for i in range(len(min_files)):

            # Short time-scale 
            day_shortts = gong_seq_trend_removed[i,:,:]
            header = fits.open(f'{year_path}/{day_folder}/{min_file}')[0].header
            hdu_short = fits.PrimaryHDU(day_shortts, header=header)
            hdu_short.writeto(f'{year_path}_shortts/{day_folder}/{min_files[i]}', overwrite=True)

            # Long time-scale 
            day_longts = gong_seq_smooth_gauss[i,:,:]
            hdu_short = fits.PrimaryHDU(day_longts, header=header)
            hdu_short.writeto(f'{year_path}_longts/{day_folder}/{min_files[i]}', overwrite=True)


else:
    
    raise ValueError(f'Task {args.task} not recognized')
    
    

print('DONE')
    