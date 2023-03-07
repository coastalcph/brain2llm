import random
from pathlib import Path

import numpy as np
import torch

from brain_LMs.LMs_extractors import LMExtractor

from .fmri_dataloader import FMRIWordLevel
from .utils.utils_helper import CV_ind


class EmbedsDictsBuilder:
    def __init__(self, config, fmri_type="token"):
        self.dataset_name = config.datasets.dataset_name
        self.model_name = config.models.model_name
        self.model_alias = config.models.model_alias
        self.model_type = config.models.model_type
        self.if_cased = "uncased" if "uncased" in self.model_name else "cased"
        self.wordlist, self.brain_wordlist = self.get_wordlist(config, fmri_type)
        self.layer_num = config.models.n_layers
        self.word_decon_save_dir = config.datasets.word_reps_dir
        self.alias_emb_dir = config.datasets.alias_emb_dir
        self.is_average = config.models.is_avg
        self.dict_path = Path(config.datasets.dict_dir) / fmri_type
        self.num_folds = config.datasets.num_folds
        self.words = list(
            dict.fromkeys(
                [
                    w.lower().split("_", 1)[0]
                    if self.if_cased == "uncased" or self.dataset_name == "pereira"
                    else w.split("_", 1)[0]
                    for w in self.wordlist
                ]
            )
        )

    @staticmethod
    def get_wordlist(config, fmri_type):
        dim_size = min(config.datasets.tr_num, config.gaussian_params.vec_dim)
        data = torch.load(
            f"{config.datasets.fmri_reps_dir}/{config.datasets.dataset_name}-{fmri_type}-sub--{config.datasets.num_subjects}-{config.gaussian_params.lookbackward}-{config.gaussian_params.lookforward}-{dim_size}.pth"
        )
        if config.datasets.dataset_name == "pereira":
            words = open(config.datasets.fmri_sentences_path).read().split()
            word_id_pairs = [
                f"{FMRIWordLevel.remove_punct(word)[0]}_{idx}"
                for idx, word in enumerate(words)
            ]
            return word_id_pairs, data["dico"]
        else:
            return data["dico"], data["dico"]

    def process_embeddings(self, config):
        if config.datasets.tr_num > config.gaussian_params.vec_dim:
            dim_list = [config.gaussian_params.vec_dim]
        else:
            dim_list = [config.datasets.tr_num, config.gaussian_params.vec_dim]
        # print(f"Dictionary size: {len(self.pairs_dict)}")
        if self.model_name == "fasttext" or self.model_alias == "openai":
            self.__process_api_embeddings(config, dim_list)
        else:
            self.__process_contextual_embeddings(config, dim_list)

    def __process_contextual_embeddings(self, config, dim_list):
        suffix = "_averaged" if self.is_average else "_first"
        word_embs_save_path = Path(
            f"{self.word_decon_save_dir}{suffix}/{self.model_name}"
        )
        word_embs_save_path.mkdir(parents=True, exist_ok=True)

        for dim_size in dim_list:
            for layer in range(self.layer_num):
                save_file_path = (
                    word_embs_save_path
                    / f"{self.dataset_name}_decon_{self.model_name}_dim_{dim_size}_layer_{layer}.pth"
                )
                if save_file_path.exists():
                    print(f"File {save_file_path} already exists. Skipping function.")
                else:
                    file_path = (
                        Path(self.alias_emb_dir)
                        / suffix[1:]
                        / self.model_name
                        / f"{self.dataset_name}_{self.model_name}_dim_{dim_size}_length_sentences_layer_{layer}.pth"
                    )
                    if not file_path.exists():
                        lm_reps = LMExtractor(config)
                        lm_reps.get_lm_rep()

                    # alias_embeddings = np.load(file_path, allow_pickle=True)
                    alias_data = torch.load(file_path)

                    word_embeddings = self.get_mean_word_embeddings(
                        alias_data, self.words, self.if_cased
                    )
                    torch.save(
                        {"dico": self.words, "vectors": word_embeddings}, save_file_path
                    )

    def __process_api_embeddings(self, config, dim_list):
        suffix = "_averaged" if self.is_average else "_first"
        word_embs_save_path = Path(
            f"{self.word_decon_save_dir}/{suffix[1:]}/{self.model_name}"
        )
        word_embs_save_path.mkdir(parents=True, exist_ok=True)
        for dim_size in dim_list:
            save_file_path = (
                word_embs_save_path
                / f"{self.dataset_name}_{self.model_name}_dim_{dim_size}_layer_0.pth"
            )
            if save_file_path.exists():
                print(f"File {save_file_path} already exists. Skipping function.")
            else:
                api_extractor = LMExtractor(config, list(self.words))
                api_extractor.get_lm_rep()

    def get_mean_word_embeddings(self, alias_data, words, cased):
        vecs = alias_data["vectors"]
        dico = np.array(alias_data["dico"])
        if cased == "uncased" or self.dataset_name == "pereira":
            dico = np.array([w.lower() for w in dico])
        word_embeddings = torch.empty((len(words), vecs.shape[1]))
        word_indices = [np.where(dico == w)[0] for w in words]
        for i, indices in enumerate(word_indices):
            word_embeddings[i] = torch.mean(vecs[indices], dim=0)
        return word_embeddings

    def build_dictionary(self, wordlist=None):
        self.dict_path.mkdir(parents=True, exist_ok=True)

        # Check if fold dictionary files exist for all folds and full_word_dict_file exists
        fold_dict_files_exist = all(
            (
                self.dict_path
                / f"{self.if_cased}_{self.model_type}_{s}_{self.dataset_name}_fold_{i}.txt"
            ).exists()
            for s in ["train", "test"]
            for i in range(self.num_folds)
        )

        full_word_dict_file = (
            self.dict_path
            / f"{self.if_cased}_{self.model_type}_all_words_{self.dataset_name}_regression_eval.txt"
        )

        if fold_dict_files_exist and full_word_dict_file.exists():
            # print(f"Number of words: {len(brain_words)}")
            print(
                "Train and test dictionaries already exist. Skipping dictionary building."
            )
            return

        shuffled_dict = self.get_words_dict(wordlist)
        shuffled_res = np.array(list(shuffled_dict.items()), dtype=object)
        ind = CV_ind(len(shuffled_dict), n_folds=self.num_folds)

        for ind_num in range(self.num_folds):
            train_ind = ind != ind_num
            test_ind = ind == ind_num

            train_alias = shuffled_res[train_ind]
            test_alias = shuffled_res[test_ind]

            train_dico = [
                (sub_val, key)
                for _, pairs in train_alias
                for src in pairs
                for key, val in src.items()
                for sub_val in val
            ]
            test_dico = [
                (sub_val, key)
                for _, pairs in test_alias
                for src in pairs
                for key, val in src.items()
                for sub_val in val
            ]

            with open(
                self.dict_path
                / f"{self.if_cased}_{self.model_type}_train_{self.dataset_name}_fold_{ind_num}.txt",
                "w",
            ) as file_writer:
                for sub_val, key in train_dico:
                    file_writer.write(f"{sub_val}    {key}\n")
                file_writer.close()
            with open(
                self.dict_path
                / f"{self.if_cased}_{self.model_type}_test_{self.dataset_name}_fold_{ind_num}.txt",
                "w",
            ) as file_writer:
                for sub_val, key in test_dico:
                    file_writer.write(f"{sub_val}    {key}\n")
                file_writer.close()

        all_dico = [
            (sub_val, key)
            for _, pairs in shuffled_res
            for src in pairs
            for key, val in src.items()
            for sub_val in val
        ]

        with open(full_word_dict_file, "w") as file_writer:
            for sub_val, k in all_dico:
                file_writer.write(f"{sub_val}    {k}\n")
            file_writer.close()

    def get_words_dict(self, wordlist):
        if wordlist is None:
            if self.if_cased == "uncased":
                brain_words = list(
                    dict.fromkeys(
                        [w.lower().split("_", 1)[0] for w in self.brain_wordlist]
                    )
                )
            else:
                brain_words = list(
                    dict.fromkeys([w.split("_", 1)[0] for w in self.brain_wordlist])
                )
            shuffled_words = random.sample(list(brain_words), len(brain_words))
            # shuffled_words = brain_words
        else:
            shuffled_words = random.sample(list(wordlist), len(wordlist))
            # shuffled_words = wordlist

        if self.if_cased == "uncased" or self.dataset_name == "pereira":
            pairs_dict = {
                w.lower(): [
                    w_id
                    for w_id in self.brain_wordlist
                    if w_id.lower().split("_", 1)[0] == w.lower()
                ]
                for w in self.words
            }
        else:
            pairs_dict = {
                w: [w_id for w_id in self.brain_wordlist if w_id.split("_", 1)[0] == w]
                for w in self.words
            }

        shuffled_dict = {}
        for key in shuffled_words:
            pair_dict = {key: pairs_dict.get(key, [])}
            key_lower = key.lower()
            values = shuffled_dict.setdefault(key_lower, [])
            if key == key_lower:
                if {key_lower: pairs_dict.get(key_lower, [])} not in values:
                    values.append({key_lower: pairs_dict.get(key_lower, [])})
            else:
                if pair_dict not in values:
                    values.append(pair_dict)

        return shuffled_dict
