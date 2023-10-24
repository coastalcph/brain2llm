#!/bin/bash
#SBATCH --job-name=nats_new
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=48GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#SBATCH --exclude=hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl
#SBATCH --array=0-5
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=48:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
declare -A SIZE=([0]=bert_uncased_L-2_H-128_A-2 [1]=bert_uncased_L-4_H-256_A-4  [2]=bert_uncased_L-4_H-512_A-8 [3]=bert_uncased_L-8_H-512_A-8  [4]=bert-base-uncased [5]=bert-large-uncased)
export size=${SIZE[$SLURM_ARRAY_TASK_ID]}

# run all bert size procrustes
python3 pipe_test_debug.py --config nat_configs/${size}.yaml --data:dataset_name new_nat_story1 --data:alias_emb_dir "/projects/nlp/data/brain/datasets/nat_stories/newexperiments/LM_outputs" --data:word_decontextualized_embs_dir "/projects/nlp/data/brain/datasets/nat_stories/newexperiments/LM_decontext_embs" --data:word_level_fmri_rep_dir '/projects/nlp/data/brain/datasets/nat_stories/experiments_debug/word_level_fmri'


