import argparse
import copy
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
import pandas as pd

from MUSE.src.evaluation import Evaluator
from MUSE.src.models import build_model
from MUSE.src.trainer import Trainer
from MUSE.src.utils import bool_flag, initialize_exp
from src.fmri_dataloader import FMRIWordLevel

class Muse:
    def __init__(self, config, train_eval):
        self.config = config
        self.num_folds = 4
        self.model_type = config.model.model_type
        self.model_name = config.model.model_name
        self.model_alias = config.model.model_alias
        self.dict_dir = config.data.dict_dir
        self.word_decon_embs_dir = Path(config.data.word_decontextualized_embs_dir)
        self.image_emb = Path(config.data.image_emb)
        self.do = train_eval
        self.src_lang = config.muse_parameters.src_lang
        self.tgt_lang = config.muse_parameters.tgt_lang

    def set_muse_param(self, image_emb_path, lm_emb_path, bin_name, fold, vec_dim):
        muse_params = copy.deepcopy(self.config.muse_parameters)
        muse_params.tgt_emb = lm_emb_path
        muse_params.emb_dim = vec_dim
        if fold is not None: # if fold is None, it means in regression evaluation
            # muse_params.dico_eval = f"{self.dict_dir}/test_{fold}{bin_name}_1k_only.txt"
            # muse_params.dico_train = f"{self.dict_dir}/train_wiki_dico_{fold}{bin_name}_1k_only.txt"
            muse_params.dico_eval = f"{self.dict_dir}/test_{fold}{bin_name}_cleaned.txt"
            muse_params.dico_train = f"{self.dict_dir}/train_wiki_dico_{fold}{bin_name}_cleaned.txt"
            muse_params.src_emb = image_emb_path
        else:
            muse_params.dico_eval = f"{self.dict_dir}/image_test_fold_{fold}{bin_name}.txt"
            muse_params.src_emb = image_emb_path

        return muse_params


    def run(self, extend_exp=None):
        exp_flag = "" if extend_exp is None else f"_{extend_exp}"
        bins = {
            "_freq": ["_freq500", "_freq5000", "_freq_end"],
            "_poly": ["_one_meaning", "_over3_meaning", "_2or3_meaning"]
        }.get(exp_flag[:5], [""])

        project_type = "procrustes" if self.do == "train" else "regression"
        project_name = f"image2{self.model_type}_{project_type}-nips-rebuttal{exp_flag}"
        wandb.init(project=project_name, name=f"{self.src_lang}_{self.tgt_lang}")
        metrics_df = pd.DataFrame()
        for dim in [128, 256, 512, 768, 1024, 1280, 1600, 2048]:
            for lm_emb_path in list(self.word_decon_embs_dir.iterdir()):
                if not lm_emb_path.name.startswith(self.model_name):
                    continue
                for image_emb_path in list(self.image_emb.iterdir()):
                    if f"{dim}.pth" in lm_emb_path.name.split("_")[-1] and f"{dim}.pth" in image_emb_path.name.split("_")[-1]:
                        metrics = {"VM": image_emb_path.name.split("_")[0],
                                   "Models": self.model_name}
                        for bin_name in bins:
                            if extend_exp is not None:
                                metrics.update({"Bins" : bin_name[1:]})
                            if self.do == "train":
                                for fold in [203, 255, 633, 813, 881]:
                                    metrics.update({"Fold": f"fold_{fold}"})

                                    muse_params = self.set_muse_param(str(image_emb_path), str(lm_emb_path), bin_name, fold, dim)

                                    muse_res = muse_supervised(muse_params)
                                    metrics.update(muse_res)
                                    # build dataframe from dictionary
                                    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
                                # else:
                                #     muse_params = self.set_muse_param(layer, sub, bin_name)
                                #     muse_res = muse_evaluate(muse_params)
                                #     metrics.update(muse_res)
                                #     # build dataframe from dictionary
                                #     metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
                            metrics.clear()
        wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=True)
        wandb.finish()


def muse_supervised(configs):
    params = argparse.Namespace(**configs)
    # check parameters
    assert not params.cuda or torch.cuda.is_available()
    assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
    assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
    assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
    assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    trainer.load_training_dico(params.dico_train)

    VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_100'
    VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_100-S2T-10000'

    # define the validation metric
    VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
    logger.info("Validation metric: %s" % VALIDATION_METRIC)

    """
    Learning loop for Procrustes Iterative Learning
    """

    n_iter = 0
    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)
    result_metrics = {
        'P@1-CSLS': to_log["precision_at_1-csls_knn_100"],
        'P@5-CSLS': to_log["precision_at_5-csls_knn_100"],
        'P@10-CSLS': to_log["precision_at_10-csls_knn_100"],
        'P@30-CSLS': to_log["precision_at_30-csls_knn_100"],
        'P@50-CSLS': to_log["precision_at_50-csls_knn_100"],
        'P@100-CSLS': to_log["precision_at_100-csls_knn_100"],
        'P@1-NN': to_log["precision_at_1-nn"],
        'P@5-NN': to_log["precision_at_5-nn"],
        'P@10-NN': to_log["precision_at_10-nn"],
        'P@30-NN': to_log["precision_at_30-nn"],
        'P@50-NN': to_log["precision_at_50-nn"],
        'P@100-NN': to_log["precision_at_100-nn"],
        "mean_cosin": to_log["mean_cosine-csls_knn_100-S2T-10000"]
    }

    return result_metrics


def muse_evaluate(configs):
    params = argparse.Namespace(**configs)
    # check parameters
    params.normalize_embeddings = ""
    assert params.src_lang, "source language undefined"
    assert os.path.isfile(params.src_emb)
    assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    # run evaluations
    to_log = OrderedDict({'n_iter': 0})
    evaluator.monolingual_wordsim(to_log)
    # evaluator.monolingual_wordanalogy(to_log)
    if params.tgt_lang:
        evaluator.crosslingual_wordsim(to_log)
        evaluator.word_translation(to_log)
        evaluator.sent_translation(to_log)
    result_metrics = {
        "precision_at_1-csls_knn_100": to_log["precision_at_1-csls_knn_100"],
        "precision_at_5-csls_knn_100": to_log["precision_at_5-csls_knn_100"],
        "precision_at_10-csls_knn_100": to_log["precision_at_10-csls_knn_100"],
        "precision_at_30-csls_knn_100": to_log["precision_at_30-csls_knn_100"],
        "precision_at_50-csls_knn_100": to_log["precision_at_50-csls_knn_100"],
        "precision_at_100-csls_knn_100": to_log["precision_at_100-csls_knn_100"],
        "precision_at_1-nn": to_log["precision_at_1-nn"],
        "precision_at_5-nn": to_log["precision_at_5-nn"],
        "precision_at_10-nn": to_log["precision_at_10-nn"],
        "precision_at_30-nn": to_log["precision_at_30-nn"],
        "precision_at_50-nn": to_log["precision_at_50-nn"],
        "precision_at_100-nn": to_log["precision_at_100-nn"],
    }
    return result_metrics

if __name__ == "__main__":
    # main
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="", help="Export embeddings after training (txt / pth)")

    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5,
                        help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_train", type=str, default="default",
                        help="Path to training dictionary (default: use identical character strings)")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_100',
                        help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0,
                        help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    parser.add_argument("--load_optim", type=bool_flag, default=False, help="Reload optimal")

    # parse parameters
    config = parser.parse_args()
    muse_supervised(config)

