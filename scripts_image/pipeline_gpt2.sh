#!/bin/bash
#SBATCH --job-name=img_gpt2
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=24GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-2
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --output=reg-%x.%j.out
#SBATCH --time=30:00:00
#SBATCH --exclude=hendrixgpu05fl

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
declare -A SIZE=([0]=gpt2 [1]=gpt2-large [2]=gpt2-xl)
export size=${SIZE[$SLURM_ARRAY_TASK_ID]}

# run all gpt2 size procrustes
python3 function_test.py --config ./image_config/${size}.yaml

# run all gpt2 size regression
#python3 pipeline.py --config ./hp_configs/${size}.yaml --method regression

