import itertools
import math
import random


def dico_build(ids_words_path, seed, id_list=None, name_list=None):
    """
    It builds a dictionary from a list of words
    
    Args:
      ids_words_path: path to the file containing the ids and words
      seed: the seed for the random number generator
      id_list: list of ids to be included in the dictionary
      name_list: list of names of the dictionaries to be built
    """
    pairs = open(ids_words_path).readlines()
    id_all = [i.strip().split(': ')[0] for i in pairs]
    name_all = [i.strip().lower().split(': ')[1] for i in pairs]

    if id_list:
        id_all = id_list
    if name_list:
        name_all = name_list
    
    related_dict = {}
    for id_v in id_all:
        related_names = []
        for idx, id_val in enumerate(id_all):
            if id_v == id_val:
                name = name_all[idx]
                related_names.append(name)
        if id_v in related_dict.keys():
            continue
        else:
            related_dict[id_v] = related_names
    id_nums = len(related_dict)
    print(id_nums)

    if seed > -1:
        keys = list(related_dict.keys())
        random.seed(seed)
        random.shuffle(keys)
        shuffled_dict = dict()
        for key in keys:
            shuffled_dict.update({key: related_dict[key]})
    else:
        shuffled_dict = related_dict

    for value in shuffled_dict.values():
        relations = list(product(value, repeat=2))
        for src, tgt in relations:
            word_dico = f'{src}    {tgt}'
            dico.append(word_dico)
    
    i = iter(shuffled_dict.items())
    train_part = dict(itertools.islice(i, math.ceil(len(shuffled_dict) * 0.7)))
    eval_part = dict(itertools.islice(i, math.ceil(len(shuffled_dict) * 0.15)))
    test_part = dict(i)
    # build id2w dict
    train_dico = []
    eval_dico = []
    test_dico = []

    for key, values in train_part.items():
        for value in values:
            train_dico.append(f'{key} {value}')

    for key, values in eval_part.items():
        for value in values:
            eval_dico.append(f'{key} {value}')

    for key, values in test_part.items():
        for value in values:
            test_dico.append(f'{key} {value}')
    print(len(train_dico), len(eval_dico), len(test_dico))
    return train_dico, eval_dico, test_dico

def dico_write(write_dir, dicos, seed):
    for idx, part in enumerate(["train","eval","test"]):
        with open(f'{write_dir}/{part}_wiki_dico_{seed}.txt', 'w+') as biw:
            biw.write('\n'.join(dicos[idx]))
            biw.close()


