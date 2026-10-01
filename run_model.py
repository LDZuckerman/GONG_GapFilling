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


def run_model(d, gpu, continue_train, test_only=False, debug=False):
    
    ###############
    # test_only = True
    ###############
    
    # Set out and data dirs
    name = d['name'] 
    outdir = "../model_runs" 
    exp_outdir = f'{outdir}/{name}/'
    
    # If continueing a previous training run, rename old files
    continue_train = False if continue_train in ['False', False] else int(continue_train)
    if continue_train: # Integars evaluate True
        print(f'Continueing training of {name} - relabeling previous outputs with previous number of epochs')
        # prev_dict = json.load(open(f'{exp_outdir}/exp_file.json','r')) 
        prev_epochs = np.sort([int(f.replace('epoch','').replace('_examples.png','')) for f in os.listdir(exp_outdir) if 'examples.png' in f])[-1] # prev_dict['num_epochs']
        add_epochs = continue_train 
        preds_files = [f for f in os.listdir(exp_outdir) if 'test_preds' in f] # for some models, have already created test preds on different datasets
        for f in preds_files:
            os.rename(f'{exp_outdir}/{f}', f'{exp_outdir}/{f}_e{prev_epochs}')
        metrics_files = [f for f in os.listdir(exp_outdir) if 'metrics' in f] # for some models, have already created test preds on different datasets
        for f in metrics_files: 
            os.rename(f'{exp_outdir}/{f}', f'{exp_outdir}/{f.replace(".pkl",f"_e{prev_epochs}.pkl")}')
        os.rename(f'{exp_outdir}/losses.npy', f'{exp_outdir}/losses_e{prev_epochs}.npy')
        os.rename(f'{exp_outdir}/{name}.pth', f'{exp_outdir}/{name}_e{prev_epochs}.pth')
    
    # Copy exp dict file for convenient future reference and create exp outdir 
    if not os.path.exists(exp_outdir): 
        print(f'Creating experiment output dir {exp_outdir}')
        os.makedirs(exp_outdir)
    elif not test_only and not continue_train:
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

    # Define model (if continueing, load back in)
    print('Loading model', flush=True)
    only_centers = False
    only_outers = False
    if d['loss_name'] in ['CtrOnly_MSE','OtrOnly_MSE']:
        print(f'NOTE -  using {d["loss_name"]}, so will also enforce predicted images to be zero {"outside" if d["loss_name"] == "CtrOnly_MSE" else "inside"} of central region')
        if d['loss_name'] == 'CtrOnly_MSE': only_centers = True 
        if d['loss_name'] == 'OtrOnly_MSE': only_outers = True 
    xs0, ys0 = next(iter(train_loader))
    model = run_utils.get_model(d, xs0, device, only_centers, only_outers)
    if continue_train != 'False' and continue_train > 0:
        model.load_state_dict(torch.load(f'{exp_outdir}/{name}_e{prev_epochs}.pth', map_location=torch.device(device), weights_only=True))
    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model all epochs
        optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
        losses = []
        if not continue_train:
            print(f"Training {name} (training on {device})", flush=True)
        else:
            print(f"Adding {add_epochs} of training time, starting at previous epoch {prev_epochs}")
        start_epoch = 0 if not continue_train else prev_epochs
        train_epochs = d['num_epochs'] if not continue_train else add_epochs 
        for epoch in range(train_epochs):
            
            # Train 
            epoch_num = start_epoch + epoch
            print(f'Epoch {epoch_num}', flush=True)
            ctr_wgt = 'NaN' if 'ctr_wgt' not in d.keys() else d['ctr_wgt']
            train_loss = run_utils.train_net(train_loader, model,  d['loss_name'], ctr_wgt, optimizer, device=device, save_dir=exp_outdir, epoch=epoch_num)
            losses.append(train_loss.detach().cpu().numpy()) 
            print('   Loss: ',train_loss.detach().cpu().numpy())
            torch.save(model.state_dict(), f'{exp_outdir}/{name}_temp.pth') # save after each epoch in case training crashes

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{name}.pth')
        print(f'Saving trained model as {exp_outdir}/{name}.pth, and saving average losses', flush=True)
        
        # Save losses
        if continue_train:
            prev_losses = np.load(f'{exp_outdir}/losses_e{prev_epochs}.npy')
            losses = np.concatenate((prev_losses, losses))
        np.save(f'{exp_outdir}/losses', losses)

    # Load it back in and save results on test data 
    model = run_utils.get_model(d, xs0, device, only_centers, only_outers)
    model.load_state_dict(torch.load(f'{exp_outdir}/{name}.pth'))
    model.eval()
    file_names = test_ds.x_sets if 'pixels' not in d['dataset'] else None
    run_utils.save_model_results(test_loader, file_names=file_names, save_dir=f'{exp_outdir}/test_preds_scale', model=model)


if __name__ == "__main__":
    
    # Read in arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=str, required=True)
    parser.add_argument("-continue_train", "--continue_train", required=False, default=False)
    args = parser.parse_args()
    
    # Test gpu
    print(args.gpu)
    if args.gpu == 'True':
        print('GPU Available:', torch.cuda.is_available(), 'Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
    
    # Iterate through experiments (or single experiment)
    with open(args.f) as file:
        d = json.load(file)
    t0 = datetime.now()
    verb = 'CONTINUEING' if eval(args.continue_train) > 0 else 'RUNNING'
    print(f'{verb} EXPERIMENT {d["name"]} \nstart time {t0}\nexp dict: {d}')
    run_model(d, args.gpu, args.continue_train)
    print(f'DONE')
    dt = datetime.now() - t0
    print(f'FINISHED EXPERIMENT (took {int(dt.total_seconds() / 60)} minutes)')

