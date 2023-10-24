import numpy as np
from pathlib import Path

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr
from src.utils.utils_helper import normalization
import pandas as pd

from src.muse_debug import Muse


class RSA(Muse):
    def __init__(self, config, fmri_type="type"):
        super().__init__(config, "type")
        self.config = config
        self.model_name = config.model.model_name
        self.model_alias = config.model.model_alias
        # self.dataset_name = "new_nat_story1"
        self.fmri_type = "type"
        self.word_emb_unique_save_dir = config.data.word_decontextualized_embs_dir
        self.layers = config.model.get("n_layers")
        self.is_average = config.model.is_avg
        self.tr_num = config.data.tr_num
        self.lm_dim_size = config.model.dim
        self.if_cased = "uncased" if "uncased" in self.model_name else "cased"
        self.suffix = "averaged" if self.is_average else "first"


    def calculate_geometry(self, sample_embeds):
        sim_mat = spearmanr(sample_embeds, axis=1)[0]
        dissim_mat = np.ones(sim_mat.shape) - sim_mat
        geometry = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
        return geometry


    def run_rsa(self):
        if self.model_name == "fasttext" or self.model_alias == "openai":
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}/{self.suffix}/{self.model_name}")
        else:
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}_{self.suffix}/{self.model_name}")

        rsa_results = []
        for layer in range(self.layers):
            if self.model_name == "fasttext" or self.model_alias == "openai":
                lm_file_path = word_embs_save_path / f"{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_layer_0.pth"
            else:
                lm_file_path = word_embs_save_path / f"{self.dataset_name}_decon_{self.model_name}_dim_{self.lm_dim_size}_layer_{layer}.pth"
            lm_data = torch.load(lm_file_path)

            for sub in range(1, self.subjects + 1):
                brain_data = torch.load(
                    self.outfile_dir / f"{self.dataset_name}-{self.fmri_type}-sub--{sub}-{self.lookback}-{self.lookout}-{self.vec_dim}.pth")

                # pdb.set_trace()
                lm_embeds = normalization(lm_data["vectors"]).numpy()
                lm_words = lm_data["dico"]
                brain_words = brain_data["dico"]
                brain_embeds = normalization(brain_data["vectors"]).numpy()

                lm_lst = [lm_words.index(w.lower()) if self.if_cased == "uncased" else lm_words.index(w) for w in
                          brain_words]
                lm_vectors = lm_embeds[lm_lst]


                lm_embeds = lm_vectors
                # brain_embeds = brain_embeds

                lm_geometry = self.calculate_geometry(lm_embeds)
                brain_geometry = self.calculate_geometry(brain_embeds)

                rho, p = spearmanr(lm_geometry, brain_geometry)
                project_name = f"{self.dataset_name}-brain2{self.model_alias}_rsa_{self.suffix}"
                wandb_tags = [f"sub_{sub}",
                              f"{self.model_name}", f"layer_{layer}"]

                wandb.init(project=project_name,
                                   name="_".join(wandb_tags),
                                   tags=wandb_tags)
                rsa_results.append((sub, layer, rho, p))
                wandb.log({
                    "rsascore": rho,
                    "pvalue": p})
                wandb.finish()
        return rsa_results

    def run_rsa_debug(self, extend_exp=None):
        exp_flag = "" if extend_exp is None else f"_{extend_exp}"
        if self.model_name == "fasttext" or self.model_alias == "openai":
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}/{self.suffix}/{self.model_name}")
        else:
            word_embs_save_path = Path(f"{self.word_emb_unique_save_dir}_{self.suffix}/{self.model_name}")
        project_name = f"{self.dataset_name}-rsa-brain2{self.model_type}_{self.suffix}{exp_flag}"
        wandb.init(project=project_name, name=f"{self.src_lang}_{self.tgt_lang}",
                   tags=[f"{self.src_lang}", f"{self.tgt_lang}"])

        metrics_df = pd.DataFrame()

        for layer in range(self.layers):
            if self.model_name == "fasttext" or self.model_alias == "openai":
                lm_file_path = word_embs_save_path / f"{self.dataset_name}_{self.model_name}_dim_{self.lm_dim_size}_layer_0.pth"
            else:
                lm_file_path = word_embs_save_path / f"{self.dataset_name}_decon_{self.model_name}_dim_{self.lm_dim_size}_layer_{layer}.pth"
            lm_data = torch.load(lm_file_path)

            for sub in range(1, self.subjects + 1):
                brain_data = torch.load(
                    self.outfile_dir / f"{self.dataset_name}-{self.fmri_type}-sub--{sub}-{self.lookback}-{self.lookout}-{self.tr_num}.pth")

                metrics = {"Subjects": f"subject-{sub}",
                           "Models": self.model_name,
                           "Layers": f"layer-{layer}"}
                # pdb.set_trace()

                # lm_embeds = normalization(lm_data["vectors"]).numpy()
                lm_embeds = lm_data["vectors"].numpy()
                lm_words = lm_data["dico"]
                brain_words = brain_data["dico"]
                brain_embeds = normalization(brain_data["vectors"]).numpy()
                lm_lst = [lm_words.index(w.lower()) if self.if_cased == "uncased" else lm_words.index(w) for w in
                          brain_words]
                lm_vectors = lm_embeds[lm_lst]

                lm_embeds = lm_vectors

                lm_geometry = self.calculate_geometry(lm_embeds)
                brain_geometry = self.calculate_geometry(brain_embeds)

                rho, p = spearmanr(lm_geometry, brain_geometry)

                metrics.update({"rsascore": rho,"pvalue": p})
                metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
                metrics.clear()
        wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=True)
        wandb.finish()
        return "finished"

