import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pickle
import os
import json
import sys 
sys.path.append('/pl/active/NSO-IT/data/leah/Solar_GapFilling/GONG_GapFilling/')
from utils import eval_utils


def read_in_files(output_dir, model_name, example_idx, add_back_long, scale_preds):
    '''
    Helper function to get inputs, preds, and trues
    '''
    
    # Get validation trues, preds, and input for given model at example idx
    shortts = 'True' if 'shortTS' in json.load(open(f'{output_dir}/{model_name}/exp_file.json'))['dataset'] else False
    shortts_str = '_shortts' if shortts else ''
    preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
    true = np.load(f'{preds_dir}/true{shortts_str}_{example_idx}.npy')
    pred = np.load(f'{preds_dir}/pred{shortts_str}_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x{shortts_str}_{example_idx}.npy')
    t = pickle.load(open(f'{preds_dir}/starttimes.pkl','rb'))[example_idx]
    s = f'{t[7:9]}:{t[9:11]} on {t[2:4]}/{t[4:6]}/20{t[0:2]}'
    
    # If desired, add back long-timescale oscilations for for short-timescale-only models
    if shortts and add_back_long:
        true = np.load(f'{preds_dir}/true_{example_idx}.npy') # use full true
        inp_long = np.load(f'{preds_dir}/true_longts_{example_idx}.npy')
        pred = pred + inp_long # reconstruct full pred by adding back in LI-predicted long-TS trends  
        
    # If desired, scale preds to the same range as inputs
    if scale_preds:
        nongap_idxs = np.where(np.sum(inp, axis=(1,2)) != 0)[0]
        inp_nongap = inp[nongap_idxs]
        inp_min = np.min(inp_nongap, axis=0)
        inp_max = np.max(inp_nongap, axis=0)
        gap_idxs = np.where(np.sum(inp, axis=(1,2)) == 0)[0]
        pred_gap = pred[gap_idxs]
        pred_min = np.min(pred_gap, axis=0)
        pred_max = np.max(pred_gap, axis=0)
        pred_scale =  inp_min + (pred - pred_min) * (inp_max -  inp_min) / (pred_max - pred_min)
        pred = pred_scale
        
    return inp, true, pred, s


def plot_preds_one_model_single(model_name, model_desc, output_dir, example_idx=0, cbar=False, show_diff=True, show_LI=True, add_back_long=True, scale_preds=False): # rescale=False
    '''
    Plot model predictions along with inputs and truth for one example image set from the validation data
    '''
    
    # Get input, true, and pred
    inp, true, pred, s = read_in_files(output_dir, model_name, example_idx, add_back_long, scale_preds)
    
    # If desired, get linear interpolation predictions for comparison
    if show_LI:
        dataset = json.load(open(f'{output_dir}/{model_name}/exp_file.json','rb'))['dataset']
        li_preds_dir = f'{output_dir}/linear_interpolation/LI_test_preds_{dataset}_new'
        if not os.path.exists(li_preds_dir):
            eval_utils.save_li_filled_set(dataset)
        li_pred = np.load(f'{li_preds_dir}/pred_{example_idx}.npy')
            
    # Re-scale if desired
    # if rescale:
    #     # mins_maxs = pickle.load(open(''))
    #     MIN, MAX = -120, 120 # mins_maxs[mins_maxs['dataset'] == dataset]['min'], mins_maxs[mins_maxs['dataset'] == dataset]['max']
    #     true = true * (MAX - MIN) + MIN
    #     pred = pred * (MAX - MIN) + MIN
    #     inp = inp * (MAX - MIN) + MIN

    # Find the missing image indices in the input in order to label
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    
    # Set up plot
    N = true.shape[0]
    gridspec = {'width_ratios': [1] * N + [0.4]}
    rows = 5 if (show_diff and show_LI) else 4 if (show_diff or show_LI) else 3
    vmin = np.min(true); vmax = np.max(true)
    fig, axs = plt.subplots(rows, N+1, figsize=(N*2, rows*2.5), gridspec_kw=gridspec)
    #fig, axs = plt.subplots(4,15, figsize=(30,10))
    plt.suptitle(f'Validation Example Sarting at {s} for {model_desc}', fontsize=28, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.01)
    
    # Plot at each index
    for i in range(N):
        
        # Show true, input, and pred
        im0 = axs[0,i].imshow(true[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[0,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        im1 = axs[1,i].imshow(inp[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[1,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        im2 = axs[2,i].imshow(pred[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[2,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        
        # Show residual, if desired
        if show_diff:
            # pred = (pred[i,:,:] - np.min(pred[i,:,:])) / (np.max(pred[i,:,:]) - np.min(pred[i,:,:]))
            im3 = axs[3,i].imshow(pred[i,:,:]-true[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[3,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
        
        # Show predictions of linear interp for reference, if desired
        if show_LI:
            li_row = 4 if show_diff else 3
            im4 = axs[li_row,i].imshow(li_pred[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[li_row,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Add red boxes around missing images
        if i in use_idxs:
            axs[0,i].set_title(f'MI {i}', color='red', fontsize=20)
            left = axs[0,i].get_position().x0
            width = (axs[0,i].get_position().x1 - left) * 1.001
            bottom = 0.05 if show_diff else 0.07
            top = 0.875 if show_diff and show_LI else 0.86 if show_diff else 0.815
            rect = patches.Rectangle((left, bottom), width, top, transform=fig.transFigure, linewidth=5, edgecolor='red', facecolor='none', zorder=10+i) # 0.845
            fig.patches.append(rect)
        
        # Add colorbars
        if i == N-1 and cbar:
            plt.colorbar(im0, ax=axs[0,N]); plt.colorbar(im1, ax=axs[1,N]); plt.colorbar(im2, ax=axs[2,N]); 
            if show_diff:
                plt.colorbar(im3, ax=axs[3,N])
            if show_LI:
                plt.colorbar(im4, ax=axs[li_row,N])
                
    # Labels            
    for i in range(rows):
        axs[i,N].set_axis_off()  #axs[i,N].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)   
    axs[0,0].set_ylabel('True', fontsize=20)
    axs[1,0].set_ylabel('Input', fontsize=20)
    axs[2,0].set_ylabel('Predicted', fontsize=20)     
    if show_diff:
        axs[3,0].set_ylabel('Predicted-True', fontsize=20) 
    if show_LI:
        axs[li_row,0].set_ylabel('Linear Interp', fontsize=20) 
    
    return fig


def plot_preds_one_model_multi(model_name, output_dir):
    '''
    Plot model predictions along with inputs and truth for multiple example image sets from the validation data
    '''
    
    preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
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


def plot_preds_multi_models(model_names, output_dir='../../model_runs/',  example_idx=0, cbar=True, add_back_long=True):
    '''
    Plot model predictions along with truth for one example image set from the validation data
    '''
    
    # Get validation trues and inp for given model at example idx
    preds_dir = f'{output_dir}/{model_names[0]}/test_preds_scale'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')
    t = pickle.load(open(f'{preds_dir}/starttimes.pkl','rb'))[example_idx]
    s = f'{t[7:9]}:{t[9:11]} on {t[2:4]}/{t[4:6]}/20{t[0:2]}'

    # Find the missing image indices in the input in order to label
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    
    # Set up plot
    N = true.shape[0]
    gridspec = {'width_ratios': [1] * N + [0.4]}
    rows = len(model_names) + 1
    vmin = np.min(true); vmax = np.max(true)
    fig, axs = plt.subplots(rows, N+1, figsize=(N*2, rows*2.5), gridspec_kw=gridspec)
    #plt.suptitle(f'Validation Example Sarting at {s} for {model_desc}', fontsize=28, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.01)
    
    # Add trues to plot
    for i in range(N):
        im0 = axs[0,i].imshow(true[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[0,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 

    # Loop over models and add preds to plot
    for j in range(len(model_names)):
        
        # Get preds for this model
        model_name = model_names[j]
        preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
        pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')

        # Plot at each index
        for i in range(N):

            # Show true, input, and pred
            im = axs[j+1,i].imshow(pred[i,:,:], cmap='gray', vmin=vmin, vmax=vmax); axs[j+1,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 

            # Add red boxes around missing images
            if i in use_idxs:
                axs[0,i].set_title(f'MI {i}', color='red', fontsize=20)
                left = axs[j+1,i].get_position().x0
                width = (axs[j+1,i].get_position().x1 - left) * 1.001
                bottom = 0.068
                top = 0.882
                rect = patches.Rectangle((left, bottom), width, top, transform=fig.transFigure, linewidth=5, edgecolor='red', facecolor='none', zorder=10+i) # 0.845
                fig.patches.append(rect)

            # Add colorbars
            if i == N-1 and cbar:
                plt.colorbar(im, ax=axs[j+1,N])
                    
        axs[j+1,0].set_ylabel(f'{model_name}', fontsize=20)  
                
    # Labels            
    for i in range(rows):
        axs[i,N].set_axis_off()  #axs[i,N].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)   
    axs[0,0].set_ylabel('True Pixel Value', fontsize=20)   
    
    return fig


def plot_predvtrue(model_name, model_desc, output_dir='../../model_runs/',  example_idx=0, r_ranges=None, colorby=None, add_back_long=True, breakout_by_r=False, scale_preds=False): # rescale=True
    '''
    Plot pred val as a func of true (using only the missing idxs)
    '''
    
    # Get input, true, and pred
    inp, true, pred, s = read_in_files(output_dir, model_name, example_idx, add_back_long, scale_preds)
    
    # # Re-scale if desired
    # if rescale:
    #     # mins_maxs = pickle.load(open(''))
    #     MIN, MAX = -120, 120 # mins_maxs[mins_maxs['dataset'] == dataset]['min'], mins_maxs[mins_maxs['dataset'] == dataset]['max']
    #     true = true * (MAX - MIN) + MIN
    #     pred = pred * (MAX - MIN) + MIN
    #     inp = inp * (MAX - MIN) + MIN

    # Find the missing image indices and remove images at non-missing idxs
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    use_pred = pred[use_idxs,:,:]
    use_true = true[use_idxs,:,:]
    
    # Plot 
    title = f'Predicted vs. True for {model_desc}'
    if breakout_by_r:
        r_ranges = [(0,0.24), (0.25,0.75), (0.75,1)] # 
        fig, axs = plt.subplots(1, len(r_ranges), figsize=(12, 4), sharey=True)
    else:
        fig, axs = plt.subplots(1, 1)
    fig = get_predvtrue_plot(fig, axs, use_pred, use_true, r_ranges, colorby, breakout_by_r, title) # MIN, MAX
    
    return fig 


def plot_predvtrue_multi_models(model_names, output_dir='../../model_runs/',  example_idx=0, colorby=None, breakout_by_r=True):
    '''
    Plot pred val as a func of true (using only the missing idxs) for multiple models
    '''
    
    # Get validation trues and inp for given model at example idx
    preds_dir = f'{output_dir}/{model_names[0]}/test_preds_scale'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')
    t = pickle.load(open(f'{preds_dir}/starttimes.pkl','rb'))[example_idx]
    s = f'{t[7:9]}:{t[9:11]} on {t[2:4]}/{t[4:6]}/20{t[0:2]}'

    # Find the missing image indices in the input and remove images at non-missing idxs
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    use_true = true[use_idxs,:,:]
    
    # Compute radius
    length = use_true.shape[0]
    n_pix = use_true.shape[1]
    zz, xx, yy = np.meshgrid(np.linspace(0, length-1, length), np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    max_r = 61 # 99 in original 215 x 215 res # approx max radius (in pix) inside which pixels are sun, not background
    
    # Set up plot
    r_ranges = [(0,0.24), (0.25,0.75), (0.75,1)]  
    fig, axs = plt.subplots(len(model_names), len(r_ranges), figsize=(len(r_ranges)*4, len(model_names)*3), sharey=True)
    
    # Loop over models and add pred vs true to plot
    for j in range(len(model_names)):
        
        # Get preds for this model
        model_name = model_names[j]
        preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
        pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
        use_pred = pred[use_idxs,:,:]
        
        # Plot predvtrue, broken out by r
        fig = get_predvtrue_plot(fig, axs[j,:], use_pred, use_true, r_ranges, colorby, breakout_by_r, title='', ax_title=True if j==0 else False)
        axs[j,0].set_ylabel(f'{model_name}', fontsize=10)  
                
    # Labels and adjustments         
    for ax in axs.flatten():
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)   
    fig.supxlabel('True Pixel Value', fontsize=12)  
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    
    return fig


def get_predvtrue_plot(fig, axs, use_pred, use_true, r_ranges=None, colorby=None, breakout_by_r=False, title='', comparison_line=True, ax_title=True): # MIN=0, MAX=1
    
    # Compute radius
    length = use_pred.shape[0]
    n_pix = use_pred.shape[1]
    zz, xx, yy = np.meshgrid(np.linspace(0, length-1, length), np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    max_r = 61 # 99 in original 215 x 215 res # approx max radius (in pix) inside which pixels are sun, not background
    
    if not breakout_by_r:

        # Get pix only at certain radius range if desired (if not, just remove pixels outside the actual sun)
        # if r_range != None:
        #     r0 = max_r * r_range[0] # pct to pix
        #     r1 = max_r * r_range[1] # pct to pix
        #     use_idxs = np.where((((xx-ctr_x)**2 + (yy-ctr_y)**2) >= r0**2) & (((xx-ctr_x)**2 + (yy-ctr_y)**2) <= r1**2))  
        # else:
        #     use_idxs = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
        use_idxs = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
        mask = np.zeros_like(use_pred)* np.NaN
        mask[use_idxs] = 1 
        use_pred = use_pred * mask
        use_true = use_true * mask

        # Color by radius if desired 
        r = r * 0.004 # note: max d = 61*2 pixels, and that equals 0.5 degree, so that's 0.004 degree/pix  # convert from pix to degrees
        c = r if colorby == 'r' else 'grey'

        # Plot scatter
        s = axs.scatter(use_true.flatten(), use_pred.flatten(), alpha=0.1, s=5, c=c)
        if colorby=='r':
            cb = plt.colorbar(s, label='Radius (Deg)')
            cb.solids.set(alpha=1)

        # Add linear fit and y=x line
        use_true = use_true[~np.isnan(use_true)]
        use_pred = use_pred[~np.isnan(use_pred)]
        slope, yint, _, _, _  = stats.linregress(use_true.flatten(), use_pred.flatten())
        xs = np.linspace(np.min(use_true), np.max(use_true), 100)
        axs.plot(xs, yint + slope*xs, alpha=0.5, linestyle='-', c='blue',label=f'Best Linear Fit (a={np.round(slope,1)}, b={np.round(yint,1)})')            
        if comparison_line:
            axs.plot(xs, xs, alpha=0.5, linestyle=':', c='red')
        axs.legend(loc='lower left')
        
        # Set labels
        axs.set_xlabel('True Pixel Value')
        axs.set_ylabel('Predicted Pixel Value')
        plt.title(title)

        # Add Pearson r and rmse
        x = 0.1; y1 = 0.9; y2 = 0.85
        axs.text(x, y1, r'$\rho$ ' + f'= {np.round(stats.spearmanr(use_true.flatten(), use_pred.flatten())[0], 2)}', transform=axs.transAxes)
        axs.text(x, y2, f'rmse = {np.round(np.sqrt(np.nanmean((use_true.flatten()-use_pred.flatten())**2)), 2)}', transform=axs.transAxes)
        
    else:
       
        # Loop over r ranges
        for i in range(len(r_ranges)):

            # Get pix only at certain radius range if desired (if not, just remove pixels outside the actual sun)
            r0 = max_r * r_ranges[i][0] # pct to pix
            r1 = max_r * r_ranges[i][1] # pct to pix
            use_idxs = np.where((((xx-ctr_x)**2 + (yy-ctr_y)**2) >= r0**2) & (((xx-ctr_x)**2 + (yy-ctr_y)**2) <= r1**2)) 
            use_pred_rr = np.copy(use_pred)
            use_true_rr = np.copy(use_true)
            mask = np.zeros_like(use_pred_rr) * np.NaN
            mask[use_idxs] = 1 
            use_pred_rr = use_pred_rr * mask
            use_true_rr = use_true_rr * mask
            #pickle.dump({'x':use_true_rr.flatten(),'y':use_pred_rr.flatten()}, open('temp_data.pkl', 'wb')); a=b

            # Plot scatter 
            alpha = 0.01 if len(use_pred) > 40 else 0.5
            s = axs[i].scatter(use_true_rr.flatten(), use_pred_rr.flatten(), alpha=alpha, s=2)
        
            # Add linear fit and y=x line
            use_true_rr = use_true_rr[~np.isnan(use_true_rr)]
            use_pred_rr = use_pred_rr[~np.isnan(use_pred_rr)]
            #lr = LinearRegression().fit(use_true_rr.flatten().reshape(-1, 1), use_pred_rr.flatten().reshape(-1, 1))
            #xs = np.linspace(np.min(use_true_rr), np.max(use_true_rr), 100)
            #axs[i].plot(xs, lr.predict(xs.reshape(-1,1)), alpha=0.5, linestyle='-', c='blue', label=f'Best Linear Fit (a={np.round(lr.coef_[0][0],1)}, b={np.round(lr.intercept_[0],1)})')
            #params, cov = curve_fit(Linear, use_true_rr.flatten(), use_pred_rr.flatten())
            #slope, yint = params[0], params[1]
            #xs = np.linspace(np.min(use_true_rr), np.max(use_true_rr), 100)
            #axs[i].plot(xs, Linear(xs, slope, yint), alpha=0.5, linestyle='-', c='blue',label=f'Best Linear Fit (a={np.round(slope,1)}, b={np.round(yint,1)})')  
            slope, yint, _, _, _  = stats.linregress(use_true_rr.flatten(), use_pred_rr.flatten())
            xs = np.linspace(np.min(use_true_rr), np.max(use_true_rr), 100)
            axs[i].plot(xs, yint + slope*xs, alpha=0.5, linestyle='-', c='blue',label=f'Best Linear Fit (a={np.round(slope,1)}, b={np.round(yint,1)})')            
            if comparison_line:
                axs[i].plot(xs, xs, alpha=0.5, c='red', linestyle=':')
            axs[i].legend(loc='lower left')
            
            # Add Pearson r and rmse
            x = 0.1; y1 = 0.9; y2 = 0.85
            axs[i].text(x, y1, r'$\rho$ ' + f'= {np.round(stats.spearmanr(use_true_rr.flatten(), use_pred_rr.flatten())[0], 2)}', transform=axs[i].transAxes)
            axs[i].text(x, y2, f'rmse = {np.round(np.sqrt(np.nanmean((use_true_rr.flatten()-use_pred_rr.flatten())**2)), 2)}', transform=axs[i].transAxes)
            
            # Label the r-range
            if ax_title:
                axs[i].set_title(f'{r_ranges[i][0]} < r < {r_ranges[i][1]}', fontsize=10)

        fig.supxlabel('True Pixel Value', y=0.001)
        ylabel = 'Residual' if 'Residual' in title else 'Predicted Pixel Value'
        y_label_pos = 0.07 
        fig.supylabel(ylabel, x=0.07)
        plt.subplots_adjust(wspace=0)
        plt.suptitle(title)

    return fig


def plot_residvtrue(model_name, model_desc, output_dir='../../model_runs/',  example_idx=0, r_range=None, colorby=None, breakout_by_r=False):
    '''
    Plot residuals as a func of true (using only the missing idxs)
    '''
    
    # Get validation trues, preds, and input for given model at example idx
    preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')
    resids = true - pred

    # Find the missing image indices and remove images at non-missing idxs
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    use_resids = resids[use_idxs,:,:]
    use_true = true[use_idxs,:,:]
    
    # Plot 
    title = f'Residuals vs. True for {model_desc}'
    fig = get_residvtrue_plot(use_resids, use_true, colorby, breakout_by_r, title) # MIN, MAX
    
    return fig 


def get_residvtrue_plot(use_pred, use_true, colorby=None, breakout_by_r=False, title=''):
    
    # Compute radius
    length = use_pred.shape[0]
    n_pix = use_pred.shape[1]
    zz, xx, yy = np.meshgrid(np.linspace(0, length-1, length), np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
    ctr_x, ctr_y = n_pix/2, n_pix/2
    r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
    max_r = 61 # 99 in original 215 x 215 res # approx max radius (in pix) inside which pixels are sun, not background
    
    if not breakout_by_r:

        # Get pix only at certain radius range if desired (if not, just remove pixels outside the actual sun)
        use_idxs = np.where(((xx-ctr_x)**2 + (yy-ctr_y)**2) <= max_r**2)
        mask = np.zeros_like(use_pred)* np.NaN
        mask[use_idxs] = 1 
        use_pred = use_pred * mask
        use_true = use_true * mask
        use_resid = use_true - use_pred
        
        # # Plot linear fit
        # hx = use_true_.flatten()[~np.isnan(use_true.flatten())]
        # hy = use_resid.flatten()[~np.isnan(use_resid.flatten())]
        # slope, yint, _, _, _  = stats.linregress(hx, hy)
        # xs = np.linspace(np.nanmin(use_true), np.nanmax(use_true), 100)
        # ax.plot(xs, yint + slope*xs, alpha=0.5, linestyle='-', c='blue',label=f'Best Linear Fit (a={np.round(slope,1)}, b={np.round(yint,1)})')  
        
        # Plot scatter
        fig, ax = plt.subplots(1, 1)
        s = ax.scatter(use_true.flatten(), use_resid.flatten(), alpha=0.1, s=5, c=r)
        cb = plt.colorbar(s, label='Radius (Pix)')
        cb.solids.set(alpha=1)

        # Set labels
        ax.set_xlabel('True Pixel Value')
        ax.set_ylabel('Residual')
        plt.title(title)

        # Add Pearson r and rmse
        x = 0.1; y1 = 0.9; y2 = 0.85
        ax.text(x, y1, r'$\rho$ ' + f'= {np.round(stats.spearmanr(use_true.flatten(), use_pred.flatten())[0], 2)}', transform=ax.transAxes)
        ax.text(x, y2, f'rmse = {np.round(np.sqrt(np.nanmean((use_true.flatten()-use_pred.flatten())**2)), 2)}', transform=ax.transAxes)
        
    else:
        
        # Define plot
        r_ranges = [(0,0.24), (0.25,0.75), (0.75,1)] # 
        fig, axs = plt.subplots(1, len(r_ranges), figsize=(12, 4), sharey=True)
        
        # Loop over r ranges
        for i in range(len(r_ranges)):

            # Get pix only at certain radius range if desired (if not, just remove pixels outside the actual sun)
            r0 = max_r * r_ranges[i][0] # pct to pix
            r1 = max_r * r_ranges[i][1] # pct to pix
            use_idxs = np.where((((xx-ctr_x)**2 + (yy-ctr_y)**2) >= r0**2) & (((xx-ctr_x)**2 + (yy-ctr_y)**2) <= r1**2)) 
            use_pred_rr = np.copy(use_pred)
            use_true_rr = np.copy(use_true)
            mask = np.zeros_like(use_pred_rr) * np.NaN 
            mask[use_idxs] = 1 
            use_pred_rr = use_pred_rr * mask
            use_true_rr = use_true_rr * mask
            use_resid_rr = use_true_rr - use_pred_rr
            
            # Plot scatter
            s = axs[i].scatter(use_true_rr.flatten(), use_resid_rr.flatten(), alpha=0.01, s=2)

            # # Plot linear fit
            # hx = use_true_rr.flatten()[~np.isnan(use_true_rr.flatten())]
            # hy = use_resid_rr.flatten()[~np.isnan(use_resid_rr.flatten())]
            # slope, yint, _, _, _  = stats.linregress(hx, hy)
            # xs = np.linspace(np.nanmin(use_true_rr), np.nanmax(use_true_rr), 100)
            # axs[i].plot(xs, yint + slope*xs, alpha=0.5, linestyle='-', c='blue', label='Linear Trend')  
            # axs[i].legend(loc='lower right')
            
        fig.supxlabel('True Pixel Value', y=0.001) 
        fig.supylabel('Residual', x=0.07)
        plt.subplots_adjust(wspace=0)
        plt.suptitle(title)

        
    return fig


def plot_epoch_examples(data, true, pred, save_dir, epoch):
    '''
    Plot examples from last training batch of given epoch
    Include all layers of input, true, and pred
    Should be useful for any model
    '''
    
    data, true, pred = data[-1,:,:,:].cpu().numpy(), true[-1,:,:,:].cpu().numpy(), pred[-1,:,:,:].cpu().numpy() # take last set from batch
    
    use_idxs = np.where(np.max(data, axis=(1,2)) == np.min(data, axis=(1,2)))[0] 

    N = data.shape[0]
    fig, axs = plt.subplots(3,N, figsize=(N,3))
    axs[0,0].set_ylabel('Input')
    axs[1,0].set_ylabel('Pred')
    axs[2,0].set_ylabel('True')
    for i in range(N):
        axs[0,i].imshow(data[i,:,:], cmap='gray'); axs[0,i].axis('off')
        axs[1,i].imshow(pred[i,:,:], cmap='gray'); axs[1,i].axis('off')
        axs[2,i].imshow(true[i,:,:], cmap='gray'); axs[2,i].axis('off')
        if i in use_idxs:
            axs[0,i].set_title(f'Idx {i}', color='red')  

    plt.title(f'Examples from epoch {epoch} last train batch')
    plt.savefig(f'{save_dir}/epoch{epoch}_examples')

    
def plot_preds_hist(model_name, model_desc, output_dir, example_idx=0): # rescale=False

    # Get validation trues, preds, and input for given model at example idx
    preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')

    # Find the missing image indices in the input in order to label
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    
    # # Re-scale if desired
    # if rescale:
    #     # mins_maxs = pickle.load(open(''))
    #     MIN, MAX = -120, 120 # mins_maxs[mins_maxs['dataset'] == dataset]['min'], mins_maxs[mins_maxs['dataset'] == dataset]['max']
    #     true = true * (MAX - MIN) + MIN
    #     pred = pred * (MAX - MIN) + MIN
    #     inp = inp * (MAX - MIN) + MIN

    # Plot hist
    missing_trues = []
    missing_preds = []
    for i in range(15):
        missing_trues.extend(true[i,:,:].flatten())
        missing_preds.extend(pred[i,:,:].flatten())
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))  
    ax.set_title(f'Histogram of Normalized Values for {model_name} Example {example_idx}')
    ax.hist(missing_trues, alpha = 0.5, label='True')
    ax.hist(missing_preds, alpha = 0.5, label='Predicted')
    xlabel = 'Values' #if rescale else 'Normalized Values'
    ax.set_xlabel(xlabel)
    plt.legend()
    
    return fig


def plot_resids_hist(model_name, model_desc, output_dir, example_idx=0): 

    # Get validation trues, preds, and input for given model at example idx
    preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
    true = np.load(f'{preds_dir}/true_{example_idx}.npy')
    pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
    inp = np.load(f'{preds_dir}/x_{example_idx}.npy')

    # Find the missing image indices in the input
    use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 

    # Plot hist
    missing_resids = []
    for i in use_idxs:
        missing_resids.extend(true[i,:,:].flatten()-pred[i,:,:].flatten())
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))  
    ax.set_title(f'Histogram of Residuals for {model_name} Example {example_idx}')
    ax.hist(missing_resids, bins=30)
    xlabel = 'Residuals' #if rescale else 'Normalized Values'
    ax.set_xlabel(xlabel)
    
    return fig


def plot_pix_signal(model_name, model_desc, output_dir, example_idx=0, add_back_long=True, pix_idx=(50,50), scale_preds=False):

    # # Get validation trues, preds, and input for given model at example idx
    # shortts = 'True' if 'shortTS' in json.load(open(f'{output_dir}/{model_name}/exp_file.json'))['dataset'] else False
    # shortts_str = '_shortts' if shortts else ''
    # preds_dir = f'{output_dir}/{model_name}/test_preds_scale'
    # true = np.load(f'{preds_dir}/true{shortts_str}_{example_idx}.npy')
    # pred = np.load(f'{preds_dir}/pred{shortts_str}_{example_idx}.npy')
    # inp = np.load(f'{preds_dir}/x{shortts_str}_{example_idx}.npy')
    # t = pickle.load(open(f'{preds_dir}/starttimes.pkl','rb'))[example_idx]
    # s = f'{t[7:9]}:{t[9:11]} on {t[2:4]}/{t[4:6]}/20{t[0:2]}'
    
    # Get input, true, and pred
    inp, true, pred, s = read_in_files(output_dir, model_name, example_idx, add_back_long, scale_preds)
    
    # If short-timescale-only model and add_back_long
    shortts = 'True' if 'shortTS' in json.load(open(f'{output_dir}/{model_name}/exp_file.json'))['dataset'] else False
    if shortts and add_back_long:
        true = np.load(f'{output_dir}/{model_name}/test_preds_scale/true_{example_idx}.npy') # use full true
        inp_long = np.load(f'{output_dir}/{model_name}/test_preds_scale/true_longts_{example_idx}.npy')
        pred = pred + inp_long # reconstruct full pred by adding back in LI-predicted long-TS trends
    
    # Get LI 
    dataset = json.load(open(f'{output_dir}/{model_name}/exp_file.json','rb'))['dataset']
    li_preds_dir = f'{output_dir}/linear_interpolation/LI_test_preds_{dataset}_new'
    if not os.path.exists(li_preds_dir):
        eval_utils.save_li_filled_set(dataset)
    li_pred = np.load(f'{li_preds_dir}/pred_{example_idx}.npy')
    
    # Plot full signal
    fig = get_pix_signal_plot(true, pred, li_pred, pix_idx, only_gap_pred=True, inp=inp)
    fig.suptitle(f'Signal for Pixel ({pix_idx[0]}, {pix_idx[1]}) of Example Set {example_idx} as Predicted by {model_name}')
    
    return fig


def get_pix_signal_plot(true, pred, li_pred, pix_idx, only_gap_pred=True, inp=None):
    
    # Get true, pred, and LI signals
    true_signal = true[:,pix_idx[0], pix_idx[1]]
    pred_signal = pred[:,pix_idx[0], pix_idx[1]]
    if not isinstance(li_pred, type(None)):
        li_signal = li_pred[:,pix_idx[0], pix_idx[1]]
    
    # Set predicted signal to true outside of gap region
    if only_gap_pred:
        if type(inp) == None:
            raise ValueError('Must add input (to determine gap idxs) if want to plot at only gap idxs')
        pred_signal = np.where(np.sum(inp, axis=(1, 2))==0, pred_signal, true_signal)
    
    # Create plot
    if len(true_signal) > 1000:
        figsize = (200, 7)
        lw = 0.5
        xlim = (0, 1440)
        alpha = 1
    else:
        figsize = (10, 3)
        lw = 1
        alpha = 0.7
        xlim = (0, len(pred))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(xlim[0],xlim[1])
    ax.plot(pred_signal, alpha=alpha, linewidth=lw, label='Model Prediction')
    #if not li_pred == None:
    if not isinstance(li_pred, type(None)):
        ax.plot(li_signal, alpha=alpha, linewidth=lw, linestyle='--', label='Linear Interpolation')
    ax.plot(true_signal, alpha=alpha, linewidth=lw, label='True')
    ax.legend()
    ax.set_ylabel('Velocity (m/s)')
    
    return fig


def plot_vt_slice(model_name, model_desc, output_dir, example_idx, slice_x=50, add_back_long=True, scale_preds=False):
    
    # Get input, true, and pred
    inp, true, pred, s = read_in_files(output_dir, model_name, example_idx, add_back_long, scale_preds)
        
    # Plot
    fig = get_vt_slice_plot(true, pred, inp, slice_x)
    plt.suptitle(f'Temporal Slice at x={slice_x} for {model_desc}', fontsize=10, y=0.92)
    
    return fig 


def get_vt_slice_plot(true, pred, inp, slice_x, add_diff=False):
    
    # Get true and pred slices
    true_slice = true[:,slice_x,:]
    pred_slice = pred[:,slice_x,:]
    
    # Set predicted signal to true outside of gap region
    pred_slice = np.where(np.sum(inp, axis=1)==0, pred_slice, true_slice)
    
    # Transpose axes for plotting
    true_slice = np.transpose(true_slice, axes=(1, 0))
    pred_slice = np.transpose(pred_slice, axes=(1, 0))
    
    # Create plot
    n_rows = 3 if add_diff else 2
    if len(true_slice) > 1000:
        figsize = (n_rows*350, 12)
    else:
        figsize = (n_rows*5, 6)
    fig, axs = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    im0 = axs[0].imshow(true_slice, vmin=np.min(true_slice), vmax=np.max(true_slice), aspect='auto')
    axs[0].set_ylabel('True'); plt.colorbar(im0, ax=axs[0], pad=0.01, label='Velocity (m/s)')
    axs[0].set_yticks([])
    im1 = axs[1].imshow(pred_slice, vmin=np.min(true_slice), vmax=np.max(true_slice), aspect='auto')
    axs[1].set_ylabel('Predicted'); plt.colorbar(im1, ax=axs[1], pad=0.01, label='Velocity (m/s)')
    axs[1].set_yticks([])
    axs[1].set_xlabel('Time (minutes)')
    plt.subplots_adjust(hspace=0.01)
    
    # Add row for diff if required
    if add_diff:
        im2 = axs[2].imshow(pred_slice - true_slice, vmin=np.min(true_slice), vmax=np.max(true_slice), aspect='auto')
        axs[2].set_ylabel('Predicred - True'); plt.colorbar(im2, ax=axs[2], pad=0.01, label='Velocity (m/s)')
        axs[2].set_yticks([])
    
    
    return fig 


def quick_plot_set(seq, title='', colorbar=True):
    
    fig, axs = plt.subplots(1, len(seq), figsize=(len(seq), 6))
    for i in range(len(seq)):
        im = axs[i].imshow(seq[i])
    axs[int(len(seq)/2)].set_title(title)
    
    if colorbar:
        plt.colorbar(im, ax=axs[-1])
    
    for ax in axs.flatten():
        ax.set_yticks([]); ax.set_xticks([])
        
    return fig   
    

# def plot_preds_one_model_single(model_name, output_dir, example_idx=0, cbar=False):
#     '''
#     Plot model predictions along with inputs and truth for one example image set from the validation data
#     '''
    
#     # Get validation trues, preds, and input for given model at example idx
#     preds_dir = f'{output_dir}/{model_name}/test_preds'
#     true = np.load(f'{preds_dir}/true_{example_idx}.npy')
#     pred = np.load(f'{preds_dir}/pred_{example_idx}.npy')
#     inp = np.load(f'{preds_dir}/x_{example_idx}.npy')

#     # Find the missing image indices in the input in order to label
#     use_idxs = np.where(np.max(inp, axis=(1,2)) == np.min(inp, axis=(1,2)))[0] 
    
#     # Plot the examples and label the missing image indices
#     fig, axs = plt.subplots(3,15, figsize=(30,7))
#     plt.suptitle(f'{model_name} validation example {example_idx}', fontsize=28, y=1.01)
#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0.0, wspace=0.01)
#     for i in range(15):
#         im0 = axs[0,i].imshow(true[i,:,:], cmap='gray'); axs[0,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
#         im1 = axs[1,i].imshow(inp[i,:,:], cmap='gray'); axs[1,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
#         im2 = axs[2,i].imshow(pred[i,:,:], cmap='gray'); axs[2,i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
#         if i in use_idxs:
#             axs[0,i].set_title(f'MI {i}', color='red', fontsize=20)
#             left = axs[0,i].get_position().x0
#             width = (axs[0,i].get_position().x1 - left) * 1.001
#             rect = patches.Rectangle((left, 0.05), width, 0.845, transform=fig.transFigure, linewidth=5, edgecolor='red', facecolor='none', zorder=10+i)
#             fig.patches.append(rect)
#         if i == 14 and cbar:
#             plt.colorbar(im0, ax=axs[0,i]); plt.colorbar(im1, ax=axs[1,i]); plt.colorbar(im2, ax=axs[2,i]); 
#     axs[0,0].set_ylabel('True', fontsize=20)
#     axs[1,0].set_ylabel('Input', fontsize=20)
#     axs[2,0].set_ylabel('Predicted', fontsize=20)     
    
#     return fig