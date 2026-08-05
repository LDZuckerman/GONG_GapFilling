#!/bin/bash
   
#SBATCH --account=ucb520_asc3 # To use additional resources
#SBATCH --output=../../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --gres=gpu:1       # num GPU to request
#SBATCH --mem=100M  # Default 25M

# Testing mode
##SBATCH --time=1:00:00 # 24:00:00
##SBATCH --ntasks=16           # number of nodes to request  
##SBATCH --partition=atesting_mi100  # amilan for cpu, aa100 for gpu
##SBATCH --qos=testing

# Normal mode
#SBATCH --time=23:00:00  #100:00:00
#SBATCH --ntasks=21           # number of tasks (?) 
#SBATCH --partition=ami100 # amilan for cpu, ami100 (not aa100) for gpu
#SBATCH --qos=gpu-normal  #long # normal for up to 24hr, long for longer (but takes a long time to get resources)
#SBATCH --gres=gpu:mi100:1

gpu=True

module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
mamba activate pytorch241_rocm61_new_new #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;;
 esac
done

echo "Running experiment with expfile $expfile"
python run_model.py -gpu $gpu -f $expfile

#####
# run from ../ with 'sbatch run_model_gpu.sh -f exp_todo/exp_file_UNet_1.json'
# 'sbatch run_model_gpu.sh -f ../model_runs/UNet12/exp_file.json'
######

