import random
import time
from pathlib import Path

import numpy as np
import torch

from brain_LMs.utils import ridge_tools_torch
from src.fmri_dataloader import FMRIWordLevel
from .utils.utils_helper import CV_ind, normalization
from scipy.stats import zscore


class RidgeRegression(FMRIWordLevel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_folds = 4
        self.model_name = config.model.model_name
        self.model_alias = config.model.model_alias
        self.alias_emb_dir = config.data.alias_emb_dir
        self.word_emb_unique_save_dir = config.data.word_decontextualized_embs_dir
        self.layers = config.model.get("n_layers")
        self.is_average = config.model.is_avg
        self.tr_num = config.data.tr_num
        self.lm_dim_size = config.convert_parameters.vec_dim
        self.if_cased = "uncased" if "uncased" in self.model_name else "cased"
        self.suffix = "averaged" if self.is_average else "first"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_regression(self, fmri_type="type"):
        regression_files_exist, preds_save_root, word_embs_save_path = self.setup_word_emb_and_regression_path(
            fmri_type)
        if regression_files_exist:
            print("Train and test dictionaries already exist. Skipping dictionary building.")
            return

        words = None
        for layer in range(self.layers):
            for sub in range(1, self.subjects + 1):
                all_preds = []
                all_words = []
                lm_data, brain_data, words, ind, corrs_lm = self.load_lm_brain_data(word_embs_save_path, layer, sub,
                                                                                    words, fmri_type)
                for ind_num in range(self.num_folds):
                    brain_train_data, brain_test_data, lm_inflate_train_label, lm_inflate_test_label, brain_test_words = self.prepare_train_test_data(
                        lm_data, brain_data, words, ind, ind_num)

                    # Do regression
                    start = time.time()
                    weights, chosen_lambdas = ridge_tools_torch.cross_val_ridge(brain_train_data,
                                                                                lm_inflate_train_label,
                                                                                n_splits=10,
                                                                                lambdas=np.array(
                                                                                    [10 ** i for i in range(-6, 10)],
                                                                                    dtype="float32"),
                                                                                method="kernel_ridge",  # 'ridge_sk',
                                                                                do_plot=False)

                    # from fmri to nlp
                    # preds = np.dot(brain_test_data.cpu().numpy(), weights.cpu().numpy())
                    preds = brain_test_data @ weights
                    corrs_lm[ind_num, :] = ridge_tools_torch.corr_torch(preds, lm_inflate_test_label)
                    all_preds.append(preds)
                    all_words.extend(brain_test_words)
                    end = time.time()
                    print(f"Fold {ind_num} time cost:", end - start)
                    del weights, lm_inflate_test_label, lm_inflate_train_label, brain_train_data, brain_test_data, preds

                self.save_predictions(preds_save_root, layer, sub, all_words, torch.vstack(all_preds).cpu(),
                                      corrs_lm.cpu())
                del brain_data, lm_data

    def setup_word_emb_and_regression_path(self, fmri_type):
        preds_save_root = Path(f"{self.word_emb_unique_save_dir}_regression-{fmri_type}") / self.model_name
        regression_files_exist = all(((
                                              preds_save_root / f'{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_layer_{layer}_sub_{sub}.pth').exists()
                                      for layer in range(self.layers) for sub in range(1, self.subjects + 1))
                                     )

        if self.model_name == "fasttext" or self.model_alias == "openai":
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}/{self.suffix}/{self.model_name}")
        else:
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}_{self.suffix}/{self.model_name}")
        word_embs_save_path.mkdir(parents=True, exist_ok=True)

        return regression_files_exist, preds_save_root, word_embs_save_path

    def load_lm_brain_data(self, word_embs_save_path, layer, sub, words, fmri_type):

        # file_path = Path(self.alias_emb_dir) / self.model_name / \
        #             f"{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_length_sentences_layer_{layer}.npy"
        # if self.dataset_name == "pereira":
        # brain_data = torch.load(
        #     self.outfile_dir / f"{self.dataset_name}-{fmri_type}-sub--{sub}-{self.lookback}-{self.lookout}-{self.tr_num}.pth")
        # else:
        brain_data = torch.load(
            self.outfile_dir / f"{self.dataset_name}-{fmri_type}-sub--{sub}-{self.lookback}-{self.lookout}-0.pth")
        # if self.dataset_name != "pereira":
        #     brain_data["vectors"] = brain_data["vectors"].to(self.device)
        # else:
        brain_data["vectors"] = normalization(brain_data["vectors"]).to(self.device)
        # else:
        # brain_data["vectors"] = torch.from_numpy(np.nan_to_num(zscore(brain_data["vectors"].numpy()))).to(self.device)

        if self.model_name == "fasttext" or self.model_alias == "openai":
            lm_file_path = word_embs_save_path / f"{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_layer_0.pth"
            lm_data = torch.load(lm_file_path)
        else:
            lm_file_path = word_embs_save_path / f"{self.dataset_name}_decon_{self.model_name}_dim_{self.lm_dim_size}_layer_{layer}.pth"
            lm_data = torch.load(lm_file_path)
        # if self.dataset_name != "pereira" and self.model_name.startswith("opt"):
        #     lm_data["vectors"] = lm_data["vectors"].to(self.device)
        # else:
        print("LMs normalization debug")
        lm_data["vectors"] = normalization(lm_data["vectors"]).to(self.device)
            # lm_data["vectors"] = lm_data["vectors"].to(self.device)
        # lm_data["vectors"] = normalization(lm_data["vectors"]).to(self.device)
        # lm_data["vectors"] = torch.from_numpy(np.nan_to_num(zscore(lm_data["vectors"].numpy()))).to(self.device)

        # else:
        #     lm_contextual_data = np.load(file_path, allow_pickle=True)
        #     lm_data = {"dico": brain_data["dico"], "vectors": torch.from_numpy(lm_contextual_data).float()}

        if words is None:
            words = list(dict.fromkeys(
                [w.lower().split('_', 1)[0] if self.if_cased == "uncased" else w.split('_', 1)[0] for w in
                 brain_data["dico"]]))

            random.seed(self.seed)
            words = random.sample(words, len(words))
            # words = random.sample(words, 180)

        num_words = len(words) if self.dataset_name != "pereira" else 179
        ind = CV_ind(num_words, n_folds=self.num_folds)
        corrs_lm = torch.zeros((self.num_folds, lm_data["vectors"].shape[1]))

        return lm_data, brain_data, words, ind, corrs_lm

    @staticmethod
    def get_lm_inflate_label(lm_data, words, brain_words, if_cased):
        if if_cased == "cased":
            dico_words = [word_id.split('_')[0] for word_id in brain_words]
        else:
            dico_words = [word_id.lower().split('_')[0] for word_id in brain_words]
        vecs = [lm_data["vectors"][lm_data["dico"].index(word)] for dico_word in dico_words for word in words if
                word == dico_word]
        # vecs = np.vstack(vecs)
        return torch.vstack(vecs).float()
        # return torch.from_numpy(vecs).float()

    def prepare_train_test_data(self, lm_data, brain_data, words, ind, ind_num):
        train_ind = ind != ind_num
        test_ind = ind == ind_num
        train_words = np.array(words)[train_ind]
        test_words = np.array(words)[test_ind]

        # Process FMRI data, split into train and test parts
        def get_split_string(if_cased):
            return lambda word: word.split('_')[0] if if_cased == "cased" else word.lower().split('_')[0]

        split_string_fn = get_split_string(self.if_cased)
        brain_train_data_indx = np.where(np.isin([split_string_fn(word) for word in brain_data["dico"]], train_words))[
            0]
        brain_test_data_indx = np.where(np.isin([split_string_fn(word) for word in brain_data["dico"]], test_words))[0]
        brain_train_data = brain_data["vectors"][brain_train_data_indx, :].float()
        brain_test_data = brain_data["vectors"][brain_test_data_indx, :].float()
        brain_test_words = np.array(brain_data["dico"])[brain_test_data_indx]
        lm_inflate_train_label = self.get_lm_inflate_label(lm_data, train_words, brain_data["dico"], self.if_cased)
        lm_inflate_test_label = self.get_lm_inflate_label(lm_data, test_words, brain_data["dico"], self.if_cased)
        return brain_train_data, brain_test_data, lm_inflate_train_label, lm_inflate_test_label, brain_test_words

    def save_predictions(self, preds_save_root, layer, sub, all_words, all_preds, corrs_lm):
        if not preds_save_root.exists():
            preds_save_root.mkdir(parents=True, exist_ok=True)
        fname = f'{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_layer_{layer}_sub_{sub}.pth'
        print(f'saving: {str(preds_save_root / fname)}')
        print(f"number of words:{len(all_words)}")

        torch.save({'dico': all_words, 'vectors': all_preds, "corr": corrs_lm}, str(preds_save_root / fname))
