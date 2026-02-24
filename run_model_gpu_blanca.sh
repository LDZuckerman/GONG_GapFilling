#!/bin/bash
   
#SBATCH --qos=blanca-nso
#SBATCH --time=24:00:00
#SBATCH --output=../../Jobs/Job-%j.out 
#SBATCH --nodes=1         
#SBATCH --mem=160G
#SBATCH --gres=gpu:1       

module load slurm/blanca
module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
conda activate pytorch241_rocm61 #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1


while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;
 esac
done

echo "Running experiment with expfile $expfile"
python run_model.py -gpu True -f $expfile

#####
# run from ../ with 'sbatch run_model_gpu_blanca.sh -f exp_todo/exp_file_UNet.json'
######
