#!/bin/bash
#SBATCH --job-name=pere_ft
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24GB
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=20:00:00
#SBATCH --output=Pro-%x.%j.out
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl


#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
#declare -A SIZE=([0]=bert_uncased_L-2_H-128_A-2 [1]=bert_uncased_L-4_H-256_A-4  [2]=bert_uncased_L-4_H-512_A-8 [3]=bert_uncased_L-8_H-512_A-8  [4]=bert-base-uncased [5]=bert-large-uncased)
#export size=${SIZE[$SLURM_ARRAY_TASK_ID]}


# run procrustes
python3 pipe_pereira_test.py --config ./pereira_configs/fasttext.yaml

# run regression
python3 pipe_pereira_test.py --config ./pereira_configs/fasttext.yaml --method regression

#for i in 0 1 2 3;
#do
#python3 MUSE/evaluate.py  \
#    --src_lang ft --tgt_lang bg --emb_dim=300 \
#    --dico_eval  /home/kfb818/projects/b2le/data/dicts/cased_LM_train_potter_fold_${i}.txt\
#    --src_emb /home/kfb818/projects/b2le/data/test_brain_data_1_with_ft_layer_0_len_sentence.pth \
#    --tgt_emb /projects/nlp/data/brain/datasets/hp_fmri/experiments/fasttext_outputs/fasttext/potter_fasttext_dim_300_layer_0.pth ;
#    if [ $? -ne 0 ]; then
#    echo "The script brain_language_nlp/predict_nlp_from_brain.py failed."
#    exit 1
#  fi
#done

#--dico_eval /home/kfb818/projects/b2le/data/dicts/cased_LM_test_potter_fold_${i}.txt\