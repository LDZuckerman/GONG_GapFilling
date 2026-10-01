import numpy as np
import cv2
import scipy.ndimage as sndi
import pandas as pd
from sklearn import preprocessing
import astropy.io.fits as fits 
import os
import shutil
import matplotlib.pyplot as plt
import glob
import skimage as sk
import skimage
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
import scipy.ndimage as sndi
from scipy import ndimage
from torch.utils.data import Dataset
import pickle
import tarfile
import glob
from scipy.fft import fftn, fftfreq, ifft, ifftn
from scipy.interpolate import interp1d


###############
# Dataset class
###############

class dataset(Dataset):
    '''
    Dataset class for loading images and labels from a directory.
    '''
    def __init__(self, set, dataset, dpath='../Data', norm='image', channels=['X'], n_classes=2, im_size=None, subset_frac=None, freq_filter=False):
        self.dpath = dpath
        self.dataset = dataset
        self.set = set
        self.shortts = True if 'shortTS' in dataset else False
        if '$' not in dataset:
            shortts_str = '_shortts' if self.shortts else ''
            all_x_sets = [f for f in np.sort(os.listdir(f'{dpath}/{dataset}')) if f'_x{shortts_str}' in f and f != 'test_tags.npy'] 
            all_y_sets = [f for f in np.sort(os.listdir(f'{dpath}/{dataset}')) if f'_true{shortts_str}' in f and f != 'test_tags.npy']  
            test_tags = np.load(f'{self.dpath}/{self.dataset}/test_tags.npy')
            if self.set == 'train':
                self.x_sets = [f'{dataset}/{all_x_sets[i]}' for i in range(len(all_x_sets)) if all_x_sets[i].replace('_x.npy', '') not in test_tags]
                self.y_sets = [f'{dataset}/{all_y_sets[i]}' for i in range(len(all_y_sets)) if all_y_sets[i].replace('_true.npy', '') not in test_tags]
            elif self.set in ['val','test']:
                self.x_sets = [f'{dataset}/{all_x_sets[i]}' for i in range(len(all_x_sets)) if all_x_sets[i].replace(f'_x{shortts_str}.npy', '') in test_tags]
                self.y_sets = [f'{dataset}/{all_y_sets[i]}' for i in range(len(all_y_sets)) if all_y_sets[i].replace(f'_true{shortts_str}.npy', '') in test_tags]
                if self.shortts:
                    self.x_sets_og = [f.replace('_shortts', '') for f in self.x_sets]
                    self.y_sets_og = [f.replace('_shortts', '') for f in self.y_sets]
                    self.x_sets_longts = [f.replace('_x', '_x_longts') for f in self.x_sets_og]
                    self.y_sets_longts = [f.replace('_true', '_true_longts') for f in self.y_sets_og]                  
        else:
            self.x_sets = []
            self.y_sets = [] 
            print(dataset.split('$'))
            for ds in dataset.split('$'):
                all_x_sets_ds = [f for f in np.sort(os.listdir(f'{dpath}/{ds}')) if '_true' not in f and f != 'test_tags.npy']
                all_y_sets_ds = [f for f in np.sort(os.listdir(f'{dpath}/{ds}')) if '_true' in f and f != 'test_tags.npy']
                test_tags_ds = np.load(f'{self.dpath}/{ds}/test_tags.npy')
                if set == 'train':
                    x_sets_ds = [f'{ds}/{all_x_sets_ds[i]}' for i in range(len(all_x_sets_ds)) if all_x_sets_ds[i].replace('_x.npy', '') not in test_tags_ds]
                    y_sets_ds = [f'{ds}/{all_y_sets_ds[i]}' for i in range(len(all_y_sets_ds)) if all_y_sets_ds[i].replace('_true.npy', '') not in test_tags_ds]
                elif set in ['val','test']:
                    x_sets_ds= [f'{ds}/{all_x_sets_ds[i]}' for i in range(len(all_x_sets_ds)) if all_x_sets_ds[i].replace('_x.npy', '') in test_tags_ds]
                    y_sets_ds = [f'{ds}/{all_y_sets_ds[i]}' for i in range(len(all_y_sets_ds)) if all_y_sets_ds[i].replace('_true.npy', '') in test_tags_ds]
                self.x_sets.extend(x_sets_ds)
                self.y_sets.extend(y_sets_ds)
                
        self.norm = norm # ALLOW AS ARGUEMENT SO CAN CHANGE LATER
        self.im_size = im_size
        self.resize = transforms.Resize(im_size, antialias=None)
        self.freq_filter = freq_filter

    def __len__(self):
        return len(self.x_sets)

    def __getitem__(self, index):

        # Get the set with missing images
        x_path = os.path.join(self.dpath, self.x_sets[index]) 
        try:
            x = np.load(x_path)
        except Exception as e:
            print(f"Error loading from file '{x_path}': {e}")
        if self.norm == 'image':
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            if (np.max(x) == np.min(x)) or (np.isnan(x).any()) or (not np.isfinite(x).all()):
                raise ValueError(f"Warning: error normalizing data from file '{x_path}'")
        if self.im_size != None:
            try:
                x = np.array(self.resize(torch.from_numpy(np.expand_dims(x, axis=0)))).squeeze()
            except Warning as e:
                print(f"Error resizing data from file '{x_path}': {e}")

        # Get the true (completed) set
        y_path = os.path.join(self.dpath, self.y_sets[index]) 
        try:
            true = np.load(y_path)
        except ValueError as e:
            print(f'Error loading file {y_path}: {e}')
        if self.im_size != None: 
            true = np.array(self.resize(torch.from_numpy(np.expand_dims(true, axis=0)))).squeeze()
        if self.freq_filter:
            true = frequency_filter(true, f_low_mHz=2.5, f_high_mHz=4.5)
        if self.norm == 'image':
            true = (true - np.min(true)) / (np.max(true) - np.min(true))
            
        # If short-timescale ony, get long-timescale and fulls
        if self.shortts and self.set in ['test', 'val']:
            
            # Get full x
            x_og_path = os.path.join(self.dpath, self.x_sets_og[index]) 
            x_og = np.load(x_og_path)
            if self.im_size != None: 
                x_og = np.array(self.resize(torch.from_numpy(np.expand_dims(x_og, axis=0)))).squeeze()
            if self.freq_filter:
                x_og = frequency_filter(x_og, f_low_mHz=2.5, f_high_mHz=4.5)
            if self.norm == 'image':
                x_og = (x_og - np.min(x_og)) / (np.max(x_og) - np.min(x_og))
            
            # Get full y
            y_og_path = os.path.join(self.dpath, self.y_sets_og[index]) 
            true_og = np.load(y_og_path)
            if self.im_size != None: 
                true_og = np.array(self.resize(torch.from_numpy(np.expand_dims(true_og, axis=0)))).squeeze()
            if self.freq_filter:
                true_og = frequency_filter(true_og, f_low_mHz=2.5, f_high_mHz=4.5)
            if self.norm == 'image':
                true_og = (true_og - np.min(true_og)) / (np.max(true_og) - np.min(true_og))
                
            # Get long-ts x
            x_longts_path = os.path.join(self.dpath, self.x_sets_longts[index]) 
            x_longts = np.load(x_longts_path)
            if self.im_size != None: 
                x_longts = np.array(self.resize(torch.from_numpy(np.expand_dims(x_longts, axis=0)))).squeeze()
            if self.freq_filter:
                x_longts = frequency_filter(x_longts, f_low_mHz=2.5, f_high_mHz=4.5)
            if self.norm == 'image':
                x_longts = (x_longts - np.min(x_longts)) / (np.max(x_longts) - np.min(x_longts))
            
            # Get full y
            y_longts_path = os.path.join(self.dpath, self.y_sets_longts[index]) 
            true_longts = np.load(y_longts_path)
            if self.im_size != None: 
                true_longts = np.array(self.resize(torch.from_numpy(np.expand_dims(true_longts, axis=0)))).squeeze()
            if self.freq_filter:
                true_longts = frequency_filter(true_longts, f_low_mHz=2.5, f_high_mHz=4.5)
            if self.norm == 'image':
                true_longts = (true_longts - np.min(true_longts)) / (np.max(true_longts) - np.min(true_longts))
              
        # Return    
        if self.shortts and self.set in ['test', 'val']:
            return x, true, x_og, true_og, x_longts, true_longts
        else:
            return x, true
    
    
class pixel_dataset(Dataset):
    '''
    Dataset class for loading pixel-wise velocity signals from a pickle file
    '''
    def __init__(self, set, dataset, dpath='../Data'):
        
        dfile = pickle.load(open(f'{dpath}/{dataset}.pkl','rb'))
        all_x = dfile['inputs']
        all_y = dfile['trues']
        test_idxs = np.load(f'{dpath}/{dataset}_test_idxs.npy')
        if set == 'train':
            self.x_sets = [all_x[i] for i in range(len(all_x)) if i not in test_idxs]
            self.y_sets = [all_y[i] for i in range(len(all_x)) if i not in test_idxs]
        elif set == 'val' or set == 'test':
            self.x_sets = [all_x[i] for i in range(len(all_x)) if i in test_idxs]
            self.y_sets = [all_y[i] for i in range(len(all_x)) if i in test_idxs]

    def __len__(self):
        return len(self.x_sets)

    def __getitem__(self, index):

        # Get signal with missing timesteps
        x = self.x_sets[index]
        x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        x = np.expand_dims(x, axis=0) # Add channels dim
        

        # Get the true (completed) signal
        y = self.y_sets[index]
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        return x, y
    

##################
# Dataset creation    
##################

def untar_files(yy, size_checks=None):
    '''
    Untar and unzip GONG files for given year, remove month folders and place all day folders in same year folder
    Results in creation of Data/Originals/Subset_20{yy} folder containing mrfqi{yy}{*mm*}{*dd*} folders 
    '''
    
    unzip_folders = np.sort([f for f in os.listdir('Data/Originals/') if '.tar' and 'i'+yy in f]) # tared month folders
    to_folder = f'Data/Originals/Subset_20{yy}'
    print(f'Untaring all the files in {to_folder}')
    #os.mkdir(to_folder)

    # check sizes (make sure transfers succeeded)
    if size_checks != None:
        any_bad = False
        for tfolder in unzip_folders:
            if os.path.getsize(f'Data/Originals/{tfolder}')/10**9 < size_checks[tfolder]:
                print(f"WARNING: {tfolder} may not have fully transfered; is only {os.path.getsize(f'Data/Originals/{tfolder}')} bytes")
                any_bad = True
        if any_bad:
            raise ValueError('At least one file size does not match expected')
        
    # Unzip each month
    for tfolder in unzip_folders:

        print(f'Folder: {tfolder}')

        # Check that its a tar folder
        if not tfolder.endswith('.gz'):#('.tar'):
            continue

        # Untar the folder 
        print(f'    Untaring {tfolder}')
        dset = tfolder[:-7]
        with tarfile.open(f'Data/Originals/{tfolder}', 'r:*') as tar:
            tar.extractall(path=f'{to_folder}/{dset}')
        
    # Remove month folder, putting all day folders in same year folder
    for month_folder in np.sort(os.listdir(to_folder)):
        for day_folder in np.sort(os.listdir(f'{to_folder}/{month_folder}')):
            shutil.move(f'{to_folder}/{month_folder}/{day_folder}', f'{to_folder}/{day_folder}')
        os.rmdir(f'{to_folder}/{month_folder}')
        
    # Remove original files
    for file in glob.glob("Data/Originals/mrfq*"):
        os.remove(file)
               
            
def save_missing_filenames(data_subset, dpath='../Data'):
    '''
    Save list of missing images and print the number missing from each day
    ''' 
    
    data_dir = f'{dpath}/Originals/{data_subset}'
    missing_all = []
    folders = np.sort([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}') and '.ipynb' not in f])
    all_timesteps = [f'{str(h).zfill(2)}{str(m).zfill(2)}' for h in range(24) for m in range(60)] # NOT just range 1400, they actually mark hours and minutes, so there is not t0060 (instead thats t0100)

    #day_dict = {} 
    count = 0
    for folder in folders:
        #files = [f for f in os.listdir(f'{data_dir}/{folder}/') if f.endswith('.fits')]
        missing = []
        for t in all_timesteps:
            file = f'{folder}t{t}.fits'
            try:
                f = fits.open(f'{data_dir}/{folder}/{file}') 
            except FileNotFoundError:
                f = fits.open(f'{data_dir}/{folder}/{file}.gz') # For some reason, some still have .gz in the name, but reading as fits still works
            if 'FILLED' in f[0].header: # file present but marked missing
                missing += [file[12:-5]] # get time from filename
                missing_all.append(file) 
        #if len(missing) > 0:
        #    day_dict[folder] = len(missing) #print(f'Day {folder} is missing {len(missing)} images')
        # print(f'{count}/{len(folders)}', end='\r'); 
            count+=1
    pickle.dump(missing_all, open(f'{data_dir}/missing_images.pkl', 'wb'))
    print(f'{data_subset} has {len(missing_all)} missing images out of {count} total')
    #pickle.dump(day_dict, open(f'{data_dir}/missing_images_by_day.pkl', 'wb'))

            

def create_dataset(from_set, from_tag, set_length, num_missing, sample_method, redo=False, dpath='..Data', fourier_filter=False, shortts=False):
    '''
    Create new dataset with image sets of given length, masking images from inputs as prescribed
    Parameters:
     - from_set = subset to take orignal data from (e.g. '2019' or '2019_shortts')
     - set_length = length to make each obs (number of images, e.g. number of minutes)
     - num_missing = number of images to have missing from each obs 
     - sample_method = 'rand' to select random gap idxs, 'cluster' to put them all at the center, or 'multilength' to place them randomly but in chunks of [2,5,10,20,30,40,50,60]
    '''
    
    #######################################################
    # Set from and to dirs, and ensure not already created
    #######################################################
    
    from_dir = f'{dpath}/Originals/{from_set}'
    shortts_tag = '_shortTS' if shortts else ''
    print(f'shortts_tag = {shortts_tag}')
    multi_gap_lengths=False if 'multi' not in sample_method else True
    year = from_set[7:12]
    if not multi_gap_lengths:
        new_name = f'NN_Data_{year}_{set_length}_{num_missing}_{sample_method}{shortts_tag}'
    else:
        new_name = f'NN_Data_{year}_{set_length}_{num_missing}_multi{sample_method}{shortts_tag}'
    new_dir = f'{dpath}/{new_name}'
    if os.path.exists(new_dir):
        if redo == True:
            print(f'Dataset {new_name} already exists, but redo = {redo} so removing {new_name} and recreating')
            shutil.rmtree(new_dir)
        elif len([f for f in os.listdir(new_dir) if '_x' in f]) != len([f for f in os.listdir(new_dir) if '_true' in f]) or len([f for f in os.listdir(new_dir) if '_x' in f]) == 0:
            print(f'Dataset {new_name} already exists, but it appears that input set calculations did not finish, so will create inputs now')
        else:
            raise ValueError(f'Dataset {new_name} already exists and seems complete, and redo = {redo}')
            
    ##############################################################################################################
    # If truth sets of given length already exist, transfer to new dir 
    # NOTE: These truths are the original truths even if saving short-scale only too, because even in that case we want to save both
    # NOTE: We will (and should!) also hit this if we began creating inputs already and then hit an error
    # NOTE: If creating a set of filtered truths, we read in the original truths here and *then* perform the filtering 
    ##############################################################################################################
    
    # Check if there are any existing datasets from the same subset with the right length truth sets that can be used
    set_dirs_len = np.sort(glob.glob(f'{dpath}/NN_Data_{year}_{set_length}')) # all set dirs that have the given length in the name
    set_dirs_len = [d for d in set_dirs_len if os.path.isdir(d)] # remove npy and pkl files from pixel-wise datasets
    have_truth_set = False
    if len(set_dirs_len) > 0: 
        if fourier_filter:
            have_truth_set = True # because will use even non-filtered set and just perform filtering here
            set_dirs_len_filt = [d for d in set_dirs_len if 'filt' in d]
            have_filt_set = len(set_dirs_len_filt) > 0
            if have_filt_set:
                truth_use_dir = set_dirs_len_filt[0] 
            else:
                truth_use_dir = set_dirs_len[0]         
        else:  
            set_dirs_len_nonfilt = [d for d in set_dirs_len if not 'filt' in d]
            have_nonfilt_set = len(set_dirs_len_nonfilt) > 0
            if have_nonfilt_set:
                truth_use_dir = str(set_dirs_len_nonfilt[0]) # [06/16/26] for some reason this is now read as bytes if not forced to string?
                have_truth_set = True
    
    # If there are, use them
    if have_truth_set: 
        
        # Get truth sets from the dir that already has them
        print(f'Already have a dataset ({truth_use_dir}) with given length timeseries. Taking truth sets from there.')
        #print(os.listdir(truth_use_dir))
        truth_sets = np.sort([f for f in os.listdir(truth_use_dir) if 'true' in f])
        
        # Create new dir (if it doesnt already exist from a crashed run)
        if not os.path.exists(new_dir): 
            os.mkdir(new_dir)
            
        # Transfer from dir that already has them to new dir (unless continuing crashed run)
        if set_dirs_len[0] != new_dir:
            for f in truth_sets:
                
                # If we want it filtered, and it is not already, apply filter and save
                if fourier_filter and 'filt' not in truth_use_dir:
                    truth_set_filt = frequency_filter(truth_set, f_low_mHz=2.5, f_high_mHz=4.5)
                    np.save(truth_set_filt, f'{set_dirs_len[0]}/{f}')
                
                # If not, copy directly 
                else:
                    shutil.copy(f'{set_dirs_len[0]}/{f}', new_dir)
                    
    ########################################   
    # If not, create them in new dataset dir
    ######################################## 
    
    else:
        print(f'Creating truth sets of length {set_length} from {from_dir}. Saving to new set dir: {new_dir}')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        save_truth_sets(set_length, save_dir=new_dir, from_path=from_dir, shortts=shortts)
        
    ###################
    # Create input sets 
    ###################
     
    print(f'Creating input sets with {num_missing} masked, sample method = {sample_method}')
    save_input_sets(new_dir, num_missing, sample_method, set_length, dpath=dpath, shortts=shortts)
    
    ##############################################################
    # If creating set with short-timescale-only inputs, save those
    ##############################################################
    
    if shortts: 
        print(f'Saving short-timescale-only versions of the trues and inputs')
        save_short_long_TS_sets(new_dir, dpath=dpath)
    
    ######################
    # Create test tag list
    ######################
    
    print('Saving test set tags')
    test_tags = []
    all_tags = np.sort([f.replace('_x.npy', '')  for f in os.listdir(new_dir) if '_x.npy' in f])
    n_test = int(len(all_tags)*0.2)
    test_tags = np.random.choice(all_tags, n_test, replace=False)
    np.save(f'{new_dir}/test_tags.npy', test_tags)
    
    print('DONE')
    
    
def save_truth_sets(set_length, save_dir, from_path, shortts=False):
    '''
    Make sets of nframes = set_length data  
    Goal: make the most possible sets that have no missing data
        - Start at beginning of day, take first nframes that have no missing data
        - If encounter missing data before getting nframes, discard that set and skip to next frame after missing data 
        - If reach the end before getting nframes, discard that set and move to next day
    NOTE: can have set spanning two days, since still consecutive
    NOTE: if creating a short-timescale only set, will save the short-timescale only images as _x and _true, 
          with the long-timescale only images as _x_longts and _true_longts (to be used in final evaluation).
    '''
    
    nframes = set_length
                   
    # Check if missing file names already saved, and if not, save
    if not os.path.exists(f'{from_path}/missing_images.pkl'):
        print(f'Missing file names not already saved for subset {from_path.split("/")[-1]} - saving now')      
        save_missing_filenames(data_subset=from_path.split('/')[-1]) # e.g. 'Subset_2024'       
    missing_filenames = pickle.load(open(f'{from_path}/missing_images.pkl', 'rb'))
    
    # If creating dataset from 2008, remove all images from the full day val sets
    #print(len(missing_filenames))
    if '2008' in from_path:
        val_set_filenames = []
        for day in ['21','22','23','24']:
            val_set_filenames.extend(np.sort(os.listdir(f'../Data/Full_Day_Val_Sets/mrfqi0809{day}')))
        missing_filenames.extend(val_set_filenames)

    # Get all the month folders in the year (subset folder)
    folders = np.sort([f for f in os.listdir(f'{from_path}') if os.path.isdir(f'{from_path}/{f}') and '.ipynb' not in f])

    # Get all the possible filenames from all the possible timesteps
    timesteps = [f'{str(h).zfill(2)}{str(m).zfill(2)}' for h in range(24) for m in range(60)] # NOT just range 1400, they actually mark hours and minutes, so there is not t0060 (instead thats t0100)
    all_filenames = [f'{folder}/{folder}t{t}.fits' for folder in folders for t in timesteps] # all filenames - each timestep for each day for each month of the year

    # [05/20/26] Initialize dict of {idx:start-time}
    # starttimes = {}
    
    # Save all possible sets
    used_files = 0
    n_discarded_good = 0 # the number of good (non-missing) images we had to discard due to running into a missing image
    while used_files < len(all_filenames):

        # Fill the next nframes filename, but if cant get to nframes before gap, start over filling after gap
        print(f'   Building set starting at idx {used_files} ({all_filenames[used_files]})', end='\r')
        set_filenames = []
        while len(set_filenames) < nframes and used_files < len(all_filenames):
            current_file = all_filenames[used_files]
            if current_file.split('/')[1] in missing_filenames: # If missing, discard and skip to next frame
                used_files += 1 
                print(f'            Not enough files in initially computed set, so discarding the {len(set_filenames)} files in this set and starting over with set building at {used_files+1}')
                n_discarded_good += len(set_filenames) # add the number that we were able to add to this set before discarding to the number of discarded good frames 
                set_filenames = []
            else: # If not missing, add to set
                set_filenames.append(current_file)
                used_files += 1

        # If not enough frames because reached the end, discard set. NOTE: this is not where too-short sets are discarded; that happens above
        if len(set_filenames) != nframes:
            print(f'            Last set does not have enough files, so discarding the {len(set_filenames)} files in this set')
            n_discarded_good += len(set_filenames) 

        # Fill npy array with the images in the set (and if shortts, another with the longts set)
        else:

            # Read all the files and enter the data
            set_data = np.zeros((nframes, 209, 209))
            if shortts:
                set_data_longts = np.zeros((nframes, 209, 209))
            for i, filename in enumerate(set_filenames):
                try:
                    f = fits.open(f'{from_path}/{filename}')
                except FileNotFoundError:
                    f = fits.open(f'{from_path}/{filename}.gz') # For some reason, some still have .gz in the name, but reading as fits still works
                set_data[i,:,:] = f[0].data
                if shortts:
                    f_long = fits.open(f'{from_path.replace("shortts", "longts")}/{filename}')
                    set_data_longts[i,:,:] = f_long[0].data
                    

            # Get the desired filename and save
            day0 = set_filenames[0][5:11]
            time0 = set_filenames[0][24:28]
            dayf = set_filenames[-1][5:11]
            timef = set_filenames[-1][24:28]
            save_name = f'{day0}t{time0}_to_{dayf}t{timef}_true.npy'.replace('.fits', '').replace('mrfqi','')
            np.save(f'{save_dir}/{save_name}', set_data)
            if shortts:
                np.save(f'{save_dir}/{save_name}'.replace('_true','_true_longts'), set_data_longts)

        print(f'   Saved {len(os.listdir(save_dir))} sets in total from {from_path}. Had to discard {n_discarded_good} good images.')

    
def save_input_sets(new_dir, num_missing, sample_method, set_length, dpath, shortts=False):
    '''
    Iterate through trues and create the inputs by removing num_missing frames from each following sample_method
    '''
    
    # Get trues
    trues = np.sort([f for f in os.listdir(f'{new_dir}') if 'true.npy' in f])
    
    # Create input for each true
    count = 0
    for true in trues:
        
        # Load true
        if os.path.exists(f'{new_dir}/{true.replace("_true", "_x")}'):
            continue
        try: # wrap in try except because for some reason it is not uncommen for there to have been an issue when the set was created
            true_set = np.load(f'{new_dir}/{true}')
        except ValueError as e: 
            print(f'         Error loading set {true}: {e}')
            print('         File possibly corrupted -  recreating truth set with hacky redo and trying again')
            subset_dir='Subset_2019' if '2019' in new_dir else 'Subset_2024' if '2024' in new_dir else ''
            hacky_truth_redo(subset_dir, redo_true=true, save_to=new_dir, set_length=set_length)
            true_set = np.load(f'{new_dir}/{true}')
        
        # Get the gap idxs
        n_missing = num_missing #int(set_length*(pct_missing/100)) # np.random.randint(int(nframes*0.1), int(nframes*0.2)+1)
        if sample_method == 'rand':
            missing_idxs = np.random.choice(set_length, n_missing, replace=False)
        elif sample_method == 'cluster':
            middle = int(set_length/2)
            start = middle - int(n_missing/2)
            missing_idxs = [start+i for i in range(n_missing)]
        elif sample_method == 'multilength':
            # If creating with random gaps of length up to a max length of num_missing
            # Lets say total missing is also still num_missing
            # This is still different from just randomly adding because doing that would rarely result in super long gaps (right?)
            # Instead of randomly assigning the gaps, randomly assign the gap lengths
            gap_lens = assign_gap_lens(num_missing)
            # Place the gaps within the set
            avail_idxs = np.linspace(1, len(true_set)-1, len(true_set)-1).tolist() # dont allow first (or last, but that is delt with below)
            missing_idxs = []
            for gap_len in gap_lens:
                possible_starts = [idx for idx in avail_idxs if (idx+gap_len+1 < set_length) and not (np.any([j not in avail_idxs for j in np.linspace(idx-1, idx+gap_len, gap_len+2)]))] # idxs within avail idxs that are at least gap_len+1 from another gap or the end
                if len(possible_starts) != 0: # In some cases, after randomly placing earlier gaps, a later one cant be placed; we *should* then redo the placement so all gaps can be placed, which must be possible, but this should occur rarely enough that just omitting the last gap should be fine
                    start_idx = np.random.choice(possible_starts)
                    use_idxs = np.linspace(start_idx, start_idx+gap_len-1, gap_len).tolist()
                    missing_idxs = missing_idxs + use_idxs
                    for x in use_idxs:
                        avail_idxs.remove(x)
        else:
            raise ValueError(f'Sample method {sample_method} not recognized')

        # Create input that is copy of true but with blanks at desired missing idxs
        input_set = np.copy(true_set)   

        for i in missing_idxs:
            input_set[int(i),:,:] = np.zeros_like(input_set[int(i),:,:])
                 
        # Save 
        input_name = true.replace('_true', '_x')
        np.save(f'{new_dir}/{true.replace("_true", "_x")}', input_set)
        print(f'       Processsed set {count} / {len(trues)}'); count+=1
        
        
def save_short_long_TS_sets(new_dir, dpath):
    '''
    Save short- and long-timescale-only sets for each input and truth set in new_dir
    '''
    
    # Loop over each true set
    trues = np.sort([f for f in os.listdir(f'{new_dir}') if 'true.npy' in f])
    for true_set_file in trues:
        
        # Load
        true_set = np.load(f'{new_dir}/{true_set_file}')
    
        # Since even these "100%" DC sets aren't truely 100%, use LI to fill 
        gap_idxs = np.where(np.sum(true_set, axis=(1,2))==0)[0]
        valid_idxs = np.where(np.sum(true_set, axis=(1,2))!=0)[0]
        gong_seq_filled_linear = true_set.copy()
        lin_interp = interp1d(valid_idxs, true_set[valid_idxs,:,:] , axis=0, kind='linear',fill_value="extrapolate") # Add fill_value="extrapolate" so that it can predict gap idxs outside of valid idxs (e.g. if there is a missing img at the start or end of the day)
        interpolated_slice = lin_interp(gap_idxs)
        gong_seq_filled_linear[gap_idxs,:,:] = interpolated_slice

        # Get smoothed 
        gong_seq_filled_smooth = gong_seq_filled_linear.copy()
        true_set_longts = ndimage.gaussian_filter1d(gong_seq_filled_linear, sigma=9, axis=2, mode='reflect')

        # Get smoothed removed
        true_set_shortts = true_set - true_set_longts

        # Put (unexpected) gaps back in to the shortts (not the longts)
        true_set_shortts[gap_idxs,:,:] = 0.0
        
        # Save
        np.save(f'{new_dir}/{true_set_file.replace(".npy","_shortts.npy")}', true_set_shortts)
        np.save(f'{new_dir}/{true_set_file.replace(".npy","_longts.npy")}', true_set_longts)
        
    # Loop over each input set
    inps = np.sort([f for f in os.listdir(f'{new_dir}') if 'x.npy' in f])
    for inp_set_file in inps:
        
        # Load
        inp_set = np.load(f'{new_dir}/{inp_set_file}')
    
        # Use LI to fill gaps
        gap_idxs = np.where(np.sum(inp_set, axis=(1,2))==0)[0]
        valid_idxs = np.where(np.sum(inp_set, axis=(1,2))!=0)[0]
        gong_seq_filled_linear = inp_set.copy()
        lin_interp = interp1d(valid_idxs, inp_set[valid_idxs,:,:] , axis=0, kind='linear',fill_value="extrapolate") # Add fill_value="extrapolate" so that it can predict gap idxs outside of valid idxs (e.g. if there is a missing img at the start or end of the day)
        interpolated_slice = lin_interp(gap_idxs)
        gong_seq_filled_linear[gap_idxs,:,:] = interpolated_slice

        # Get smoothed 
        gong_seq_filled_smooth = gong_seq_filled_linear.copy()
        inp_set_longts = ndimage.gaussian_filter1d(gong_seq_filled_linear, sigma=9, axis=0, mode='reflect')

        # Get smoothed removed
        inp_set_shortts = inp_set - inp_set_longts

        # Put (unexpected) gaps back in to the shortts (not the longts)
        inp_set_shortts[gap_idxs,:,:] = 0.0
        
        # Save
        np.save(f'{new_dir}/{inp_set_file.replace(".npy","_shortts.npy")}', inp_set_shortts)
        np.save(f'{new_dir}/{inp_set_file.replace(".npy","_longts.npy")}', inp_set_longts)
        
                
def assign_gap_lens(num_missing):

    gap_lens = []
    while sum(gap_lens) < num_missing:
        gap_lens.append(np.random.choice([2,5,10,20,30,40,50,60]))
    if sum(gap_lens) > num_missing: # if the last entry added more than what was left
        gap_lens = gap_lens[:-1]
    
    return gap_lens

        
def hacky_truth_redo(subset_dir, redo_true, save_to, set_length):
    
    day0 = redo_true[0:6]
    time0 = redo_true[7:11]
    day1 = redo_true[15:21]
    time1 = redo_true[22:26]
    if day0 != day1: # if not, need to actually do the checking for the end of the folder, but for this one its the same
        raise Error('Since day0 != day1, need to actually do the checking for the end of the folder!')
    folder = f'mrfqi{day0}' 
    set_filenames = np.sort([f for f in os.listdir(f'../Data/Originals/{subset_dir}/{folder}') if int(time0) <= int(f[12:16]) <= int(time1)])
    set_data = np.zeros((set_length, 209, 209))
    for i, filename in enumerate(set_filenames):
        f = fits.open(f'../Data/Originals/{subset_dir}/{folder}/{filename}')
        set_data[i,:,:] = f[0].data
    np.save(f'{save_to}/{redo_true}', set_data)

        
def frequency_filter(imgset, f_low_mHz=2.5, f_high_mHz=4.5):
    '''
    Fourier transform (in time dim), filter, and inverse transform the image set
    Args:
        - imgset: should be e.g. np.load(f'Data/NN_Data_2019_20_50_cluster/190101t0621_to_190101t0640_true.npy')
    '''

    FT = fftn(imgset, axes=[0])
    freqs_mHz  = fftfreq(imgset.shape[0], d=60) * 1e3 
    mask = ( # Bandpass mask — symmetric in positive/negative frequencies
           ((freqs_mHz >=  f_low_mHz) & (freqs_mHz <=  f_high_mHz)) |
           ((freqs_mHz <= -f_low_mHz) & (freqs_mHz >= -f_high_mHz)))
    FT_filt = FT * mask[:, np.newaxis, np.newaxis] 
    true_filt = np.real(ifft(FT_filt, axis=0)).astype(np.float32)
    
    return true_filt
    
    
#######
# Misc
######

def check_inputs(train_ds, train_loader, savefig=False, name=None):
    '''
    Check data is loaded correctly
    '''
    
    # Examine train data
    print('Train data:')
    print(f'\t{len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_input, train_labels = next(iter(train_loader)) 
    in_shape = train_input.size()
    print(f'\tEach image has shape {in_shape}')
    in_layers = in_shape[1]
    N1 = in_shape[2]; N2 = in_shape[3]
    print(f'\tEach batch has data of shape {train_input.size()}, e.g. {in_shape[0]} images, {[N1, N2]} pixels each, {in_layers} layers (features)')
    #print(f'\tInput values have range ({min(train_input)}, {max(train_input)})')
    
    # Check gaps
    
    
    # Plot fig if desired
    if savefig:
        n_col = in_layers #+ out_layers
        N = 20
        fig, axs = plt.subplots(N, n_col, figsize=(n_col*4, N*4))
        for i in range(N):
            X = next(iter(train_loader)) # X, y = next(iter(train_loader)) # next batch
            for j in range(in_layers):
                ax = axs[i,j] if n_col > 1 else axs[i]
                im = ax.imshow(X[0,j,:,:]); plt.colorbar(im, ax=ax) # vmin=0, vmax=1 # ith img in batch, jth channel
                ax.set_title(f'input layer {j}')
        model_dir = 'model_runs_seg' if ('WNet' in name or 'TrNet' in name) else 'model_runs_dec' if 'dec' in name else 'model_runs_enc' if 'enc' in name else 'model_runs_mag'
        plt.savefig(f'../{model_dir}/{name}/traindata_{name}')
  




        
# def save_truth_sets_OrigLongShortTS(set_length, save_dir, from_path):
#     '''
#     Make sets of nframes = set_length data  
#     Goal: make the most possible sets that have no missing data
#         - Start at beginning of day, take first nframes that have no missing data
#         - If encounter missing data before getting nframes, discard that set and skip to next frame after missing data 
#         - If reach the end before getting nframes, discard that set and move to next day
#     In addition to saving each set from the original sequence, compute the long-TS-only and short-TS-only sequences and save corresponding sets from those as well
#     '''
    
#     nframes = set_length
                   
#     # Check if missing file names already saved, and if not, save
#     if not os.path.exists(f'{from_path}/missing_images.pkl'):
#         print(f'Missing file names not already saved for subset {from_path.split("/")[-1]} - saving now')      
#         save_missing_filenames(data_subset=from_path.split('/')[-1]) # e.g. 'Subset_2024'       
#     missing_filenames = pickle.load(open(f'{from_path}/missing_images.pkl', 'rb'))
    
#     # If creating dataset from 2008, remove all images from the full day val sets
#     #print(len(missing_filenames))
#     if '2008' in from_path:
#         val_set_filenames = []
#         for day in ['21','22','23','24']:
#             val_set_filenames.extend(np.sort(os.listdir(f'../Data/Full_Day_Val_Sets/mrfqi0809{day}')))
#         missing_filenames.extend(val_set_filenames)

#     # Get all the month folders in the year (subset folder)
#     folders = np.sort([f for f in os.listdir(f'{from_path}') if os.path.isdir(f'{from_path}/{f}') and '.ipynb' not in f])

#     # Get all the possible filenames from all the possible timesteps
#     timesteps = [f'{str(h).zfill(2)}{str(m).zfill(2)}' for h in range(24) for m in range(60)] # NOT just range 1400, they actually mark hours and minutes, so there is not t0060 (instead thats t0100)
#     all_filenames = [f'{folder}/{folder}t{t}.fits' for folder in folders for t in timesteps] # all filenames - each timestep for each day for each month of the year

#     # [05/20/26] Initialize dict of {idx:start-time}
#     # starttimes = {}
    
#     # Save all possible sets
#     used_files = 0
#     n_discarded_good = 0 # the number of good (non-missing) images we had to discard due to running into a missing image
#     while used_files < len(all_filenames):

#         # Fill the next nframes filename, but if cant get to nframes before gap, start over filling after gap
#         print(f'   Building set starting at idx {used_files} ({all_filenames[used_files]})', end='\r')
#         set_filenames = []
#         while len(set_filenames) < nframes and used_files < len(all_filenames):
#             current_file = all_filenames[used_files]
#             if current_file.split('/')[1] in missing_filenames: # If missing, discard and skip to next frame
#                 used_files += 1 
#                 print(f'            Not enough files in initially computed set, so discarding the {len(set_filenames)} files in this set and starting over with set building at {used_files+1}')
#                 n_discarded_good += len(set_filenames) # add the number that we were able to add to this set before discarding to the number of discarded good frames 
#                 set_filenames = []
#             else: # If not missing, add to set
#                 set_filenames.append(current_file)
#                 used_files += 1

#         # If not enough frames because reached the end, discard set. NOTE: this is not where too-short sets are discarded; that happens above
#         if len(set_filenames) != nframes:
#             print(f'            Last set does not have enough files, so discarding the {len(set_filenames)} files in this set')
#             n_discarded_good += len(set_filenames) 

#         # Fill npy array with the images in the set
#         else:

#             # Read all the files and enter the data
#             set_data = np.zeros((nframes, 209, 209))
#             for i, filename in enumerate(set_filenames):
#                 try:
#                     f = fits.open(f'{from_path}/{filename}')
#                 except FileNotFoundError:
#                     f = fits.open(f'{from_path}/{filename}.gz') # For some reason, some still have .gz in the name, but reading as fits still works
#                 set_data[i,:,:] = f[0].data
#                 #starttimes[IDX] = GET_FROM_HEADER_MAYBE

#             # Get the desired filename and save
#             day0 = set_filenames[0][5:11]
#             time0 = set_filenames[0][24:28]
#             dayf = set_filenames[-1][5:11]
#             timef = set_filenames[-1][24:28]
#             save_name = f'{day0}t{time0}_to_{dayf}t{timef}_true.npy'.replace('.fits', '').replace('mrfqi','')
#             np.save(f'{save_dir}/{save_name}', set_data)

#        print(f'   Saved {len(os.listdir(save_dir))} sets in total from {from_path}. Had to discard {n_discarded_good} good images.')
        

        
# def create_dataset_shortts(from_set, from_tag, set_length, num_missing, sample_method, redo=False, dpath='..Data', fourier_filter=False, multi_gap_lengths=False):
#     '''
#     Create new dataset with long-timescale oscilations removed from the image sets
#     Different from method for creating a dataset without long time-scale oscilations removed ('create_dataset'):
#         * Must add the gaps to the full input day BEFORE the long TS oscilation removal (otherwise not analogous to processing of real input days)
#         * 
#     Parameters:
#      - from_set = subset to take orignal data from (e.g. '2019' or '2019_shortts')
#        NOTE: '_shortts' indcates a set with long timescale oscillations (from supergranulation) removed, in order to focus on short timescale oscilations
#      - set_length = length to make each obs (number of images, e.g. number of minutes)
#      - num_missing = number of images to have missing from each obs 
#      - sample_method = 'rand' to select random gap idxs, 'cluster' to put them all at the center, or 'multilength' to place them randomly but in chunks of [2,5,10,20,30,40,50,60]
#     '''
    
#     #######################################################
#     # Set from and to dirs, and ensure not already created
#     #######################################################
    
#     from_dir = f'{dpath}/Originals/{from_set}'
#     shortts = '_shortts' if from_set.endswith('_shortts') else ''
#     year = from_tag if 'shortts' not in from_tag else from_tag.replace('_shortts','') 
#     if not multi_gap_lengths:
#         new_name = f'NN_Data_{year}_{set_length}_{num_missing}_{sample_method}{shortts}'
#     else:
#         new_name = f'NN_Data_{year}_{set_length}_{num_missing}_multi{sample_method}{shortts}'
#     new_dir = f'{dpath}/{new_name}'
#     if os.path.exists(new_dir):
#         if redo == True:
#             print(f'Dataset {new_name} already exists, but redo = {redo} so removing {new_name} and recreating')
#             shutil.rmtree(new_dir)
#         elif len([f for f in os.listdir(new_dir) if '_x' in f]) != len([f for f in os.listdir(new_dir) if '_true' in f]) or len([f for f in os.listdir(new_dir) if '_x' in f]) == 0:
#             print(f'Dataset {new_name} already exists, but it appears that input set calculations did not finish, so will create inputs now')
#         else:
#             raise ValueError(f'Dataset {new_name} already exists and seems complete, and redo = {redo}')
            
#     ##############################################################################################################
#     # If truth sets of given length already exist, transfer to new dir 
#     # NOTE: We will (and should!) also hit this if we began creating inputs already and then hit an error
#     # NOTE: If creating a set of filtered truths, we read in the original truths here and *then* perform the filtering 
#     ##############################################################################################################
    
#     # See if there are any existing datasets from the same subset with the right length truth sets that can be used
#     set_dirs_len = np.sort(glob.glob(f'{dpath}/NN_Data_{year}_{set_length}*{shortts}')) # all set dirs that have the given length in the name
#     set_dirs_len = [d for d in set_dirs_len if os.path.isdir(d)] # remove npy and pkl files from pixel-wise datasets
#     have_truth_set = False
#     if len(set_dirs_len) > 0: 
#         if fourier_filter:
#             have_truth_set = True # because will use even non-filtered set and just perform filtering here
#             set_dirs_len_filt = [d for d in set_dirs_len if 'filt' in d]
#             have_filt_set = len(set_dirs_len_filt) > 0
#             if have_filt_set:
#                 truth_use_dir = set_dirs_len_filt[0] 
#             else:
#                 truth_use_dir = set_dirs_len[0]         
#         else:  
#             set_dirs_len_nonfilt = [d for d in set_dirs_len if not 'filt' in d]
#             have_nonfilt_set = len(set_dirs_len_nonfilt) > 0
#             if have_nonfilt_set:
#                 truth_use_dir = str(set_dirs_len_nonfilt[0]) # [06/16/26] for some reason this is now read as bytes if not forced to string?
#                 have_truth_set = True
    
#     # If there are, use them
#     if have_truth_set: 
        
#         # Get truth sets from the dir that already has them
#         print(f'Already have a dataset ({truth_use_dir}) with given length timeseries. Taking truth sets from there.')
#         #print(os.listdir(truth_use_dir))
#         truth_sets = np.sort([f for f in os.listdir(truth_use_dir) if 'true' in f])
        
#         # Create new dir (if it doesnt already exist from a crashed run)
#         if not os.path.exists(new_dir): 
#             os.mkdir(new_dir)
            
#         # Transfer from dir that already has them to new dir (unless continuing crashed run)
#         if set_dirs_len[0] != new_dir:
#             for f in truth_sets:
                
#                 # If we want it filtered, and it is not already, apply filter and save
#                 if fourier_filter and 'filt' not in truth_use_dir:
#                     truth_set_filt = frequency_filter(truth_set, f_low_mHz=2.5, f_high_mHz=4.5)
#                     np.save(truth_set_filt, f'{set_dirs_len[0]}/{f}')
                
#                 # If not, copy directly 
#                 else:
#                     shutil.copy(f'{set_dirs_len[0]}/{f}', new_dir)
                    
#     ########################################   
#     # If not, create them in new dataset dir
#     ######################################## 
    
#     else:
#         print(f'Creating truth sets of length {set_length} from {from_dir}. Saving to new set dir: {new_dir}')
#         if not os.path.exists(new_dir):
#             os.mkdir(new_dir)
#         save_truth_sets(set_length, save_dir=new_dir, from_path=from_dir)
        
#     ######################################
#     # Create input sets and save test tags
#     ######################################
     
#     # Create inputs, saving in the new dir
#     print(f'Creating input sets with {num_missing} masked, sample method = {sample_method}')
#     save_input_sets(new_dir, num_missing, sample_method, set_length, dpath=dpath)
    
#     # Create test tag list
#     print('Saving test set tags')
#     test_tags = []
#     all_tags = np.sort([f.replace('_x.npy', '')  for f in os.listdir(new_dir) if '_x' in f])
#     n_test = int(len(all_tags)*0.2)
#     test_tags = np.random.choice(all_tags, n_test, replace=False)
#     np.save(f'{new_dir}/test_tags.npy', test_tags)
    
#     print('DONE')
