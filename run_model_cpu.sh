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
#SBATCH --partition=acpu  # amilan for cpu, aa100 for gpu
#SBATCH --qos=cpu-normal

gpu=False

module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
conda activate pytorch241_rocm61_new #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1

while getopts "f:c:" flag; do
 case $flag in
   f) expfile=$OPTARG;;
   c) continue_train=$OPTARG;;
 esac
done

echo "Running experiment with expfile $expfile and gpu=$gpu (continue_train=$continue_train)" 
python run_model.py -gpu $gpu -f $expfile -continue_train $continue_train

#####
# sbatch run_model_cpu.sh -f exp_todo/exp_file_UNet_0.json -c False
# sbatch run_model_cpu.sh -f ../model_runs/UNet46/exp_file.json -c False
# sbatch run_model_cpu.sh -f ../model_runs/UNet42/exp_file.json -c 10
######
