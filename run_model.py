import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append('Solar_GapFilling/Solar_GapFilling/Utils')
from utils import run_utils, data_utils, models
import argparse
import json
import os, shutil

# run from Solar_Gap_Filling with e.g. 'python run_model.py -gpu 0  -f exp_todo/exp_file_simpleRNN.json'

def run_model(d, gpu, test_only=False):
    
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
    im_size = 128
    train_ds = data_utils.dataset(set='train', norm='image', im_size=im_size) 
    train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
    test_ds = data_utils.dataset(set='val', norm='image', im_size=im_size) 
    test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False)
    data_utils.check_inputs(train_ds, train_loader, savefig=False, name=name)
    
    # Define model
    device = torch.device('cuda') if eval(str(gpu)) else torch.device('cpu') #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs0, ys0 = next(iter(train_loader))
    if d['model_name'] == 'SimpleNet':
        model = models.SimpleNet(len_set=xs0.shape[1], k_size=d['kernel_size'], padding_mode=d['padding_mode']).to(device)
    elif d['model_name'] == 'SimpleRNN':
        model = models.SimpleRNN(n_pix=xs0.shape[2], hidden_channels=[16, 32], num_layers=2).to(device)
    elif d['model_name'] == 'UNet':
        model = models.UNet(len_set=xs0.shape[1]).to(device)
    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model
        optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
        losses = []
        print(f"Training {name} (training on {device})", flush=True)
        for epoch in range(d['num_epochs']):
            print(f'Epoch {epoch}', flush=True)
            loss = run_utils.train_net(train_loader, model,  d['loss_name'], optimizer, device=device, save_dir=exp_outdir, epoch=epoch)
            losses.append(loss.detach().cpu().numpy())

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{name}.pth')
        print(f'Saving trained model as {exp_outdir}/{name}.pth, and saving average losses', flush=True)
        np.save(f'{exp_outdir}/losses', losses)

    # Load it back in and save results on test data 
    if d['model_name'] == 'SimpleNet':
        model = models.SimpleNet(len_set=xs0.shape[1], k_size=d['kernel_size'], padding_mode=d['padding_mode']).to(device)
    elif d['model_name'] == 'SimpleRNN':
        model = models.SimpleRNN(n_pix=xs0.shape[2], hidden_channels=[16, 32], num_layers=2).to(device)
    elif d['model_name'] == 'UNet':
        model = models.UNet(len_set=xs0.shape[1])
    model.load_state_dict(torch.load(f'{exp_outdir}/{name}.pth'))
    run_utils.save_model_results(test_loader, save_dir=f'{exp_outdir}/test_preds' , model=model)


if __name__ == "__main__":
    # Read in arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=str, required=True)
    args = parser.parse_args()
    
    # Iterate through experiments (or single experiment)
    print(os.getcwd())
    print(os.listdir('.'))
    with open(args.f) as file:
        exp_dicts = json.load(file)
    if isinstance(exp_dicts, dict): # If experiment file is a single dict, not a list of dicts
        exp_dicts = [exp_dicts]
    for d in exp_dicts:
        print(f'RUNNING EXPERIMENT {d["name"]} \nexp dict: {d}')
        run_model(d, args.gpu)
        print(f'DONE')
    print('FINISHED ALL EXPERIMENTS')

