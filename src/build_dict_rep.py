import os.path
from itertools import product
import numpy as np
import torch
import random
from .utils import io_util
from .utils.utils_helper import enforce_reproducibility
from brain_LMs.utils import utils_b2l

def process_lm_word_embeddings(model_name, embeddings, alias, layer_num, lm_word_embs_save_dir, is_average=True):

    words = list(dict.fromkeys([i.split('_', 1)[0] for i in alias]))
    w_dict = {w_unique: [alias.index(w_id) for w_id in alias if w_unique == w_id.split('_', 1)[0]] for w_unique in words}
    lm_word_embeddings = np.zeros((len(words), embeddings.shape[1]))

    for idx, (key, val) in enumerate(w_dict.items()):
        lm_word_embeddings[idx] = np.mean([embeddings[emb_ind] for emb_ind in val], axis=0)

    if is_average:
        word_embs_save_path = f"{lm_word_embs_save_dir}_averaged/{model_name}"
    else:
        word_embs_save_path = f"{lm_word_embs_save_dir}_first/{model_name}"

    if not os.path.exists(word_embs_save_path):
        os.makedirs(word_embs_save_path)

    with open(f"{word_embs_save_path}/{model_name}_{layer_num}.txt", 'w') as file_writer:
        file_writer.write(f"{lm_word_embeddings.shape[0]} {lm_word_embeddings.shape[1]}\n")
        for word, vec in zip(w_dict.keys(), lm_word_embeddings):
            file_writer.write(
                f"{word} {' '.join([str(v) for v in vec.tolist()])}\n")
        file_writer.close()
    # np.save(f'/home/kfb818/projects/b2le/debugs_data/LMs_wordlevel_embs/{model_name}',lm_word_embeddings)

def build_dictionaries(alias, dict_path, num_folds):

    words = list(dict.fromkeys([i.split('_', 1)[0] for i in alias]))
    print(len(words))
    w_dict = {w_unique: [w_id for w_id in alias if w_unique == w_id.split('_', 1)[0]] for w_unique in words}

    random.shuffle(words)
    shuffled_dict = dict()
    for key in words:
        shuffled_dict.update({key: w_dict[key]})

    ind = utils_b2l.CV_ind(len(shuffled_dict), n_folds=num_folds)
    shuffled_res = np.array(list(shuffled_dict.items()), dtype=object)

    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
    for ind_num in range(num_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num

        train_alias = shuffled_res[train_ind]
        test_alias = shuffled_res[test_ind]

        train_dico = [f"{src}    {key}" for key, val in train_alias for src in val]
        test_dico = [f"{src}    {key}" for key, val in test_alias for src in val]

        with open(f"{dict_path}/decon_train_fold_{ind_num}.txt", 'w') as file_writer:
            file_writer.write('\n'.join(train_dico))
            file_writer.close()
        with open(f"{dict_path}/decon_test_fold_{ind_num}.txt", 'w') as file_writer:
            file_writer.write('\n'.join(test_dico))
            file_writer.close()


def main():

    enforce_reproducibility(seed=42)
    parser = io_util.create_args_parser()
    config, unknown = parser.parse_known_args()
    args = io_util.load_config(config, unknown)

    lm_name = args.model.model_name
    lm_word_emb_save_path = args.data.lm_word_save_path
    fmri_word_level_path = args.data.fmri_word_level_path
    dict_root_path = args.data.dict_root_path
    fmri_word_level = open(fmri_word_level_path).readlines()
    wordlist = [i.split(' ', 1)[0] for i in fmri_word_level[1:]]

    # build_strict_dictionaries(words, dict_root_path)
    # build_flexible_dictionaries(words, dict_root_path)
    # build_decon_dictionaries(words, dict_root_path)

    for layer in range(args.model.n_layers):
        lm_embeddings_path = f"{args.data.lm_all_embeddings_root}/{args.model.model_name}/{args.model.model_name}_length_sentences_layer_{layer}.npy"
        lm_embeddings = np.load(lm_embeddings_path, allow_pickle=True)
        # build_lm_word_embeddings(lm_name, lm_embeddings, words, layer, lm_word_emb_save_path)
        process_lm_word_embeddings(lm_name, lm_embeddings, wordlist, layer, lm_word_emb_save_path)


if __name__ == "__main__":
    main()