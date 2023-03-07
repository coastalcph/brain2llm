import json
import os


def get_polyseme(args):
    """
    Given a word, return a list of all its polysemes
    
    Args:
      args: a list of arguments
    """
    polyseme_source_path = args.data.setdefault(
        "poly_src_path", 
        "/image/nlp-datasets/***/cached_requests_omw")
    cache = json.load(open(polyseme_source_path))
    words_list = open(
        args.data.get(
            "ordered_wordlist_path", 
            "./data/ordered_wordlist.txt")).read().strip().lower().split('\n')
    print('length of words_list: ', len(words_list))
    eng_table = cache['en']
    num_meaning = {}
    for word in words_list:
        if len(word.split('_')) > 1:
            num_meaning[word] = 'phrase' 
        elif word not in eng_table:
            num_meaning[word] = -1
        else:
            num_meaning[word] = eng_table[word][0]
    return num_meaning

def build_polyseme_dict(num_meaning, args, seeds):
    """
    Given a list of words, a list of seeds, and a number of meanings, build a dictionary of polysemes
    
    Args:
        num_meaning: the number of meanings you want to generate
        args: a dictionary of parameters
        seeds: a list of words that you want to use as the seed words for the polysemes.
    """
    save_dir = args.data.setdefault("polysemy_dict_dir", './data/dictionaries_polysemy')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dict_path = args.data.get("origin_dict_dir", "./data/origin_dict")
    dict_list = os.listdir(dict_path)
    # seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        bin1 = []
        bin2_3 = []
        bin_over_3 = []
        bin_phrase = []
        bin_unk = []
        bins = [bin1, bin2_3, bin_over_3, bin_phrase, bin_unk]
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()

                # eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]

                for idx, word in enumerate(eval_words):
                    if num_meaning[word] == "phrase":
                        bins[3].append(dictionaries[idx])
                    elif num_meaning[word] == -1:
                        bins[4].append(dictionaries[idx])
                    elif num_meaning[word] > 3:
                        bins[2].append(dictionaries[idx])
                    elif num_meaning[word] == 1:
                        bins[0].append(dictionaries[idx])
                    else:
                        bins[1].append(dictionaries[idx])

                print(len(bin1), len(bin2_3), len(bin_over_3), len(bin_phrase), len(bin_unk))
                for block_idx, block_name in enumerate(['single', '2to3', 'over3', 'phrase','unk']):
                    with open(f'{save_dir}/eval_{seed}_{block_name}.txt', 'w') as wd:
                        for pair in bins[block_idx]:
                            wd.write(pair)
                        wd.close()

                


