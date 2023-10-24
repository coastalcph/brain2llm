#!/bin/bash
#SBATCH --job-name=pere_opt
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=48GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=3
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --output=pro-%x.%j.out
#SBATCH --time=30:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl


#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
declare -A SIZE=([0]=opt-125m [1]=opt-1.3b [2]=opt-6.7b  [3]=opt-30b)
export size=${SIZE[$SLURM_ARRAY_TASK_ID]}

# run all opt size procrustes
#python3 pipe_pereira_test.py --config ./pereira_configs/${size}.yaml --data:word_level_fmri_rep_dir '/projects/nlp/data/brain/datasets/pereira_fmri/experiments/word_level_fmri_pca';
python3 pipe_test_debug.py --config ./pereira_configs/${size}.yaml
# run all opt size regression
#python3 pipe_pereira_test.py --config ./pereira_configs/${size}.yaml --method regression

