import numpy as np
import cv2
import sunpy
import scipy.ndimage as sndi
import pandas as pd
from sklearn import preprocessing
import astropy.io.fits as fits 
import os
import matplotlib.pyplot as plt
import skimage as sk
import skimage
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
import scipy.ndimage as sndi
from torch.utils.data import Dataset


###############
# Dataset class
###############

class dataset(Dataset):
    '''
    Dataset class for loading images and labels from a directory.
    '''
    def __init__(self, set, dpath='../Data/NN_Data', norm='image', channels=['X'], n_classes=2, im_size=None):
        self.dpath = dpath
        all_x_sets = [f for f in os.listdir(f'{self.dpath}') if '_true' not in f and f != 'test_tags.npy'] 
        all_y_sets = [f for f in os.listdir(f'{self.dpath}') if '_true' in f and f != 'test_tags.npy']
        test_tags = np.load(f'{self.dpath}/test_tags.npy')
        if set == 'train':
            self.x_sets = [all_x_sets[i] for i in range(len(all_x_sets)) if all_x_sets[i].replace('_x.npy', '') not in test_tags]
            self.y_sets = [all_y_sets[i] for i in range(len(all_y_sets)) if all_y_sets[i].replace('_true.npy', '') not in test_tags]
        elif set == 'val' or set == 'test':
            self.x_sets = [all_x_sets[i] for i in range(len(all_x_sets)) if all_x_sets[i].replace('_x.npy', '') in test_tags]
            self.y_sets = [all_y_sets[i] for i in range(len(all_y_sets)) if all_y_sets[i].replace('_true.npy', '') in test_tags]
        self.norm = norm # ALLOW AS ARGUEMENT SO CAN CHANGE LATER
        self.im_size = im_size
        self.resize = transforms.Resize(im_size, antialias=None)

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
        if self.norm == 'image':
            true = (true - np.min(true)) / (np.max(true) - np.min(true))

        return x, true
    


def check_inputs(train_ds, train_loader, savefig=False, name=None):
    '''
    Check data is loaded correctly
    '''
    print('Train data:')
    print(f'\t{len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_input, train_labels = next(iter(train_loader)) 
    in_shape = train_input.size()
    print(f'\tEach image has shape {in_shape}')
    in_layers = in_shape[1]
    N1 = in_shape[2]; N2 = in_shape[3]
    print(f'\tEach batch has data of shape {train_input.size()}, e.g. {in_shape[0]} images, {[N1, N2]} pixels each, {in_layers} layers (features)')
    

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
