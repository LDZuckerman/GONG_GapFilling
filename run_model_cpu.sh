#!/bin/bash
   
#SBATCH --account=ucb520_asc3 # To use additional resources
#SBATCH --output=../../Jobs/Job-%j.out  #../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  

# Testing mode
##SBATCH --time=1:00:00 # 24:00:00
##SBATCH --ntasks=16           # number of nodes to request  
##SBATCH --partition=atesting_mi100  # amilan for cpu, aa100 for gpu
##SBATCH --qos=testing

# Normal mode
#SBATCH --time=24:00:00 # 4:00:00
#SBATCH --ntasks=20           # number of nodes to request  
#SBATCH --partition=amilan  # amilan for cpu, aa100 for gpu
#SBATCH --qos=normal

gpu=False

module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
conda activate pytorch241_rocm61_new #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;
 esac
done

echo "Running experiment with expfile $expfile and gpu=$gpu " 
python run_model.py -gpu $gpu -f $expfile

#####
# run from ../ with 'sbatch run_model_cpu.sh -f exp_todo/exp_file_WNet.json'
######
