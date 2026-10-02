NOTE: this is an ongoing project and is still under development.

## Motivation

Using supervised machine learning, we hope to fill gaps in time-series GONG image sets. This will enable more accurate computation of solar far-side maps. 

## Approach

Model training can be run using the following bash command, where the training parameters should be set in the specified exp_file_UNet.json:

```
sbatch run_model_cpu.sh
    -f exp_todo/exp_file_UNet.json
    -c False
```

The broad strokes of model training are as follows:

* run_model_cpu.sh runs the run_model.py script, passing it the -f flag specifying the experiment dictionary to use, and the -c flag specifying whether this is a conitnuation of a previous run, and if so, how many more epochs to add. 
* The run_model() function in run_model.py gets the data specified in the experiment file, and initializes the model, optimizer and loss. For each epoch, the training is done by calling train_net() in run_utils.py.
* The train_net() function iterates through the training data loader, gets model predictions on each batch, and computes the loss.
* Back in run_model() in run_model.py, we call save_model_results() in run_utils.py to iterate through the validation set loader and save model predictions.
* The notebook at analysis/Explore_Results.ipynb can be used to examine the results of all trained models. 
