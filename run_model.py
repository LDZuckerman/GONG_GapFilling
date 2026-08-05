import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import sys
import argparse
import json, pickle
import os, shutil
from datetime import datetime
sys.path.append('Solar_GapFilling/Solar_GapFilling/Utils')
from utils import run_utils, data_utils, models


def run_model(d, gpu, test_only=False, debug=False):
    
    # Set out and data dirs
    name = d['name'] 
    outdir = "../model_runs" 
    exp_outdir = f'{outdir}/{name}/'
    
    # Copy exp dict file for convenient future reference and create exp outdir 
    if not os.path.exists(exp_outdir): 
        print(f'Creating experiment output dir {exp_outdir}')
        os.makedirs(exp_outdir)
    elif not test_only:
        print(f'Experiment output dir {exp_outdir} already exists - contents will be overwritten')
    print(f'Copying exp dict into {exp_outdir}exp_file.json')
    json.dump(d, open(f'{exp_outdir}/exp_file.json','w'))
    
    # Get data
    print(f"Loading data", flush=True)
    if 'pixel' not in d["dataset"]:
        im_size = 128
        freq_filter = False if 'freq_filter' not in d.keys() else eval(d['freq_filter']) # backwards compatability 
        train_ds = data_utils.dataset(dataset=d["dataset"], set='train', norm='image', im_size=im_size, freq_filter=freq_filter)#, subset_frac=d['subset_frac']) 
        train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
        test_ds = data_utils.dataset(dataset=d["dataset"], set='val', im_size=im_size, freq_filter=freq_filter, norm=False) # DONT NORMALIZE VAL SET
        test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False) # DONT SHUFFLE VAL SET
        data_utils.check_inputs(train_ds, train_loader, savefig=False, name=name)
    else:
        train_ds = data_utils.pixel_dataset(dataset=d["dataset"], set='train')#, subset_frac=d['subset_frac']) 
        train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
        test_ds = data_utils.pixel_dataset(dataset=d["dataset"], set='val') 
        test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False) # DONT SHUFFLE VAL SET

    # Specify device
    if gpu == 'True':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise ValueError('GPU specified but not available')
    else:   
        device = torch.device('cpu')

    # Define model
    xs0, ys0 = next(iter(train_loader))
    model = run_utils.get_model(d, xs0, device)
    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model all epochs
        optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
        losses = []
        print(f"Training {name} (training on {device})", flush=True)
        for epoch in range(d['num_epochs']):
            
            # Train 
            print(f'Epoch {epoch}', flush=True)
            ctr_wgt = 'NaN' if 'ctr_wgt' not in d.keys() else d['ctr_wgt']
            train_loss = run_utils.train_net(train_loader, model,  d['loss_name'], ctr_wgt, optimizer, device=device, save_dir=exp_outdir, epoch=epoch)
            losses.append(train_loss.detach().cpu().numpy()) 
            print('   Loss: ',train_loss.detach().cpu().numpy())
            torch.save(model.state_dict(), f'{exp_outdir}/{name}_temp.pth') # save after each epoch in case training crashes

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{name}.pth')
        print(f'Saving trained model as {exp_outdir}/{name}.pth, and saving average losses', flush=True)
        np.save(f'{exp_outdir}/losses', losses)

    # Load it back in and save results on test data 
    model = run_utils.get_model(d, xs0, device)
    model.load_state_dict(torch.load(f'{exp_outdir}/{name}.pth'))
    file_names = test_ds.x_sets if 'pixels' not in d['dataset'] else None
    run_utils.save_model_results(test_loader, file_names=file_names, save_dir=f'{exp_outdir}/test_preds_scale', model=model)


if __name__ == "__main__":
    # Read in arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=str, required=True)
    args = parser.parse_args()
    
    # Test gpu
    if args.gpu == 'True':
        print('GPU Available:', torch.cuda.is_available(), 'Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
    
    # Iterate through experiments (or single experiment)
    with open(args.f) as file:
        exp_dicts = json.load(file)
    if isinstance(exp_dicts, dict): # If experiment file is a single dict, not a list of dicts
        exp_dicts = [exp_dicts]
    for d in exp_dicts:
        t0 = datetime.now()
        print(f'RUNNING EXPERIMENT {d["name"]} \nstart time {t0}\nexp dict: {d}')
        run_model(d, args.gpu)
        print(f'DONE')
        dt = datetime.now() - t0
        print(f'FINISHED EXPERIMENT (took {int(dt.total_seconds() / 60)} minutes)')

