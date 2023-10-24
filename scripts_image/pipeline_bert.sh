#!/bin/bash
#SBATCH --job-name=img_bert
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=32GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-5
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=48:00:00
#SBATCH --output=Pro-%x.%j.out
#SBATCH --exclude=hendrixgpu05fl

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
declare -A SIZE=([0]=bert_uncased_L-2_H-128_A-2 [1]=bert_uncased_L-4_H-256_A-4  [2]=bert_uncased_L-4_H-512_A-8 [3]=bert_uncased_L-8_H-512_A-8  [4]=bert-base-uncased [5]=bert-large-uncased)
export size=${SIZE[$SLURM_ARRAY_TASK_ID]}

# run all bert size procrustes
python3 function_test.py --config ./image_config/${size}.yaml

# run all bert size regression
#python3 function_test.py --config ./hp_configs/${size}.yaml --method regression