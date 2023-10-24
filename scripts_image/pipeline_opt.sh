#!/bin/bash
#SBATCH --job-name=img_opt
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-2
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --output=reg-%x.%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
declare -A SIZE=([0]=opt-125m [1]=opt-6.7b  [2]=opt-30b)
export size=${SIZE[$SLURM_ARRAY_TASK_ID]}

# run all opt size procrustes
python3 function_test.py --config ./image_config/${size}.yaml

# run all opt size regression
#python3 pipeline.py --config ./hp_configs/${size}.yaml --method regression

