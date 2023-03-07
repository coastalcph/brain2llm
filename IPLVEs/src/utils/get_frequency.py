import json
import os

import numpy as np


def get_frequency_rank(args):
    """
    Get the rank of frequency
    
    Args:
      args: a list of arguments
    """
    word2freq_path = args.data.setdefault(
        "freq_src_path", 
        "/image/nlp-datasets/***/freq_wordlist_nltk")
    word2freq = json.load(open(word2freq_path))
    words_list = open(
        args.data.get(
            "ordered_wordlist_path", 
            "./data/ordered_wordlist.txt")).read().strip().lower().split('\n')
    print('length of words_list: ', len(words_list))
    all_words = list(word2freq.keys())
    all_freqs = [word2freq[w] for w in all_words]
    freq_sort = np.argsort(all_freqs)[::-1]
    all_words_sorted = [all_words[i] for i in freq_sort]
    word2rank = {word: i for i, word in enumerate(all_words_sorted)}

    eval_word_rank = {}
    for word in words_list:
        if word not in word2rank:
            eval_word_rank[word] = len(word2rank) + 1
        else:
            eval_word_rank[word] = word2rank[word]
    return eval_word_rank

def build_frequency_dict(eval_word_rank, args, seeds):
    """
    Create various evaluation dictionaries per seed based on frequency.
    
    Args:
      eval_word_rank: a dictionary mapping words to their ranks in the evaluation set
      args: a dictionary of parameters
      seeds: a list of words that are used to seed the word2vec model
    """
    save_dir = args.data.setdefault("freq_dict_dir", './data/dictionaries_freq')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dict_path = args.data.get("origin_dict_dir", "./data/origin_dict")
    dict_list = os.listdir(dict_path)
    # seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        # bin500 = []
        bin5000 = []
        bin50000 = []
        bin_other = []
        bin_phrase = []
        bins = [bin5000, bin50000, bin_other, bin_phrase]
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()

                # eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]

                for idx, word in enumerate(eval_words):
                    if len(word.split('_')) > 1:
                        bins[3].append(dictionaries[idx])
                    # elif eval_word_rank[word] < 500:
                    #     bins[0].append(dictionaries[idx])
                    elif 5000 > eval_word_rank[word]:
                        bins[0].append(dictionaries[idx])
                    elif 5000 <= eval_word_rank[word] < 50000:
                        bins[1].append(dictionaries[idx])
                    elif 50000 <= eval_word_rank[word]:
                        bins[2].append(dictionaries[idx])                  

                print(len(bin5000), len(bin50000), len(bin_other), len(bin_phrase))
                for block_idx, block_name in enumerate(['5k', '50k', 'others', 'phrase']):
                    with open(f'{save_dir}/eval_{seed}_{block_name}.txt', 'w') as wd:
                        for pair in bins[block_idx]:
                            wd.write(pair)
                        wd.close()

                


