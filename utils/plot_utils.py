import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plot_preds_one_model_single(model_name, output_dir, example_idx=0):
    '''
    Plot model predictions along with inputs and truth for one example image set from the validation data
    '''
    
    # Get validation trues, preds, and input for given model at example idx
    preds_dir = f'{output_dir}/{model_name}/test_preds'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')

    # Find the missing image indices in the input in order to label
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    
    # Plot the examples and label the missing image indices
    fig, axs = plt.subplots(3,15, figsize=(30,7))
    plt.suptitle(f'{model_name} validation example {example_idx}', fontsize=28, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.01)
    for i in range(15):
        axs[0,i].imshow(true[i,:,:], cmap='gray'); axs[0,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        axs[1,i].imshow(inp[i,:,:], cmap='gray'); axs[1,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        axs[2,i].imshow(pred[i,:,:], cmap='gray'); axs[2,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        if i in use_idxs:
            axs[0,i].set_title(f'MI {i}', color='red', fontsize=20)
            left = axs[0,i].get_position().x0
            width = (axs[0,i].get_position().x1 - left) * 1.001
            rect = patches.Rectangle((left, 0.05), width, 0.845, transform=fig.transFigure, linewidth=5, edgecolor='red',facecolor='none', zorder=10+i)
            fig.patches.append(rect)
    axs[0,0].set_ylabel('True', fontsize=20)
    axs[1,0].set_ylabel('Input', fontsize=20)
    axs[2,0].set_ylabel('Predicted', fontsize=20)     
    
    return fig


def plot_preds_one_model_multi(model_name, output_dir):
    '''
    Plot model predictions along with inputs and truth for multiple example image sets from the validation data
    '''
    
    preds_dir = f'{output_dir}/{model_name}/test_preds'
    N = 3
    fig, axs = plt.subplots(2*N, 15, figsize=(15, N*3))
    for i in range(N): 
        axs[2*i, 6].set_title(f'Example {i}', fontsize=14)
        n_tot = len([file for file in os.listdir(preds_dir) if 'true' in file])
        idx = np.random.randint(0, n_tot)
        true = np.load(f'{preds_dir}/true_{idx}.npy')
        pred = np.squeeze(np.load(f'{preds_dir}/pred_{idx}.npy'))
        for j in range(15):
            axs[2*i,j].imshow(true[j,:,:], cmap='gray')
            axs[2*i+1,j].imshow(pred[j,:,:], cmap='gray')
            axs[2*i+1,j].axis('off'); axs[2*i,j].axis('off')
            axs[2*i,0].set_ylabel('Input')
            axs[2*i+1,0].set_ylabel('Pred')
    plt.suptitle(f'{model_name}\n', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_epoch_examples(data, true, pred, save_dir, epoch):
    '''
    Plot examples from last training batch of given epoch
    Include all layers of input, true, and pred
    Should be useful for any model
    '''
    
    data, true, pred = data[-1,:,:,:].cpu().numpy(), true[-1,:,:,:].cpu().numpy(), pred[-1,:,:,:].cpu().numpy() # take last set from batch
    
    use_idxs = np.where(np.max(data, axis=(1,2)) == np.min(data, axis=(1,2)))[0] 

    fig, axs = plt.subplots(3,15, figsize=(15,3))
    axs[0,0].set_ylabel('Input')
    axs[1,0].set_ylabel('Pred')
    axs[2,0].set_ylabel('True')
    for i in range(15):
        axs[0,i].imshow(data[i,:,:], cmap='gray'); axs[0,i].axis('off')
        axs[1,i].imshow(pred[i,:,:], cmap='gray'); axs[1,i].axis('off')
        axs[2,i].imshow(true[i,:,:], cmap='gray'); axs[2,i].axis('off')
        if i in use_idxs:
            axs[0,i].set_title(f'Idx {i}', color='red')  

    plt.title(f'Examples from epoch {epoch} last train batch')
    plt.savefig(f'{save_dir}/epoch{epoch}_examples')