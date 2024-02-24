from pathlib import Path

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import hydra

from src.LMs_dict_rep import EmbedsDictsBuilder
from src.fmri_dataloader import FMRIWordLevel
from src.muse import Muse
from src.ridge_regression import RidgeRegression
from src.rsa import RSA
from src.utils import utils_helper
from src.config import RunConfig, MODEL_CONFIGS


cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)
for model in MODEL_CONFIGS:
    cs.store(group="models", name=f"{model}", node=MODEL_CONFIGS[model])

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: RunConfig) -> None:
    OmegaConf.resolve(cfg)
    # print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(cfg)}{'-'*20}\n")
    utils_helper.enforce_reproducibility(seed=cfg.muse_params.seed)
    method = cfg.projection_method

    load_fmri_data = FMRIWordLevel(cfg)
    load_fmri_data.fmri_data_init()
    print("-" * 25 + "convert completed" + "-" * 25)

    if cfg.models.model_type == "LM":
        lm_emb_dict_builder = EmbedsDictsBuilder(cfg)
        lm_emb_dict_builder.build_dictionary()
        print("-" * 25 + "build bi-modal dictionaries completed!" + "-" * 25)
        lm_emb_dict_builder.process_embeddings(cfg)
        print("-" * 25 + "Extract and Decontextualize LMs representation completed!" + "-" * 25)
    
    return 0
    if method == "Procrustes":
        procrustes_exp = Muse(cfg, train_eval="train")
        procrustes_exp.run()
    elif method == "rsa":
        rsa_exp = RSA(cfg)
        print(rsa_exp.run_rsa())
        print("-" * 25 + "RSA Done!" + "-" * 25)
    else:
        regression = RidgeRegression(cfg)
        regression.run_regression()
        print("-" * 25 + "Regression Done!" + "-" * 25)
        eval_exp = Muse(cfg, train_eval="eval")
        eval_exp.run()

    # exp_dir.mkdir(parents=True, exist_ok=True)
    # suffix = "_averaged" if config.models.is_avg else "_first"
    # io_util.save_config(config,
    #                     exp_dir / f'exp_{config.datasets.dataset_name}_config{suffix}_{config.models.model_name}.yaml')


if __name__ == '__main__':
    main()

