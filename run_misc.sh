#!/bin/bash
   
#SBATCH --account=ucb520_asc3 # To use additional resources
#SBATCH --time=23:59:59 # 24:00:00
#SBATCH --output=../../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=300G   #160G          # memory to request
#SBATCH --qos=normal  #long
#SBATCH --partition=amilan  # amilan for cpu, ami100 for gpu 

#SBATCH --ntasks=16

# module load anaconda/2020.11
# conda activate torchenv_new

module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
conda activate pytorch241_rocm61 #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1


while getopts "t:s:l:n:m:r:h:d:" flag; do
 case $flag in
   t) task=$OPTARG;;
   s) subset_folder=$OPTARG;;
   l) set_length=$OPTARG;;
   n) num_missing=$OPTARG;;
   m) sample_method=$OPTARG;;
   r) redo=$OPTARG;;
   h) shortts=$OPTARG;;
   d) debug_mock=$OPTARG;;
 esac
done


cmd="python run_misc.py -t $task"

[[ -n $subset_folder ]] && cmd="$cmd -s $subset_folder"
[[ -n $set_length ]] && cmd="$cmd -l $set_length"
[[ -n $num_missing ]] && cmd="$cmd -n $num_missing"
[[ -n $sample_method ]] && cmd="$cmd -m $sample_method"
[[ -n $shortts ]] && cmd="$cmd -m $shortts"
[[ -n $redo ]] && cmd="$cmd -r $redo"
#[[ -n $debug_mock ]] && cmd="$cmd -d $debug_mock"

$cmd #e.g. python run_misc.py -t $task -s $subset_folder -l $set_length -n $num_missing -m $sample_method -r $redo 

############################################################################
# E.g.
# sbatch run_misc.sh -t create_dataset -s Subset_2019 -l 40 -n 20 -m cluster -h True -r True 
# sbatch run_misc.sh -t create_dataset -s Subset_2019 -l 120 -n 60 -m multilength -h False -r True 
# sbatch run_misc.sh -t create_dataset -s Subset_2019 -l 80 -n 40 -m cluster -h False -r True
# sbatch run_misc.sh -t Save_Short_Timescale_Subsets -s None -l None -n None -m None -r None
# sbatch run_misc.sh -t create_pixel_dataset -s None -l None -n None -m None -h False -r None
# sbatch run_misc.sh -t predict_validation_days_LI -s None -l None -n None -m None -r None
# sbatch run_misc.sh -t zip_files -s None -l None -n None -m None -r None
# sbatch run_misc.sh -t redo_vals -s None -l None -n None -m None -r None
## DOES NOT WORK YET ## sbatch run_misc.sh -t create_dataset -s Subset_2018$Subset_2019$Subset_2020 -l 20 -p 50 -m cluster -r True 
############################################################################

