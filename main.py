import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.config import MODEL_CONFIGS, RunConfig
from src.fmri_dataloader import FMRIWordLevel
from src.LMs_dict_rep import EmbedsDictsBuilder
from src.muse import Muse
from src.ridge_regression import RidgeRegression
from src.rsa import RSA
from src.utils import utils_helper

cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)
for model in MODEL_CONFIGS:
    cs.store(group="models", name=f"{model}", node=MODEL_CONFIGS[model])


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: RunConfig) -> None:
    OmegaConf.resolve(cfg)
    print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(cfg)}{'-'*20}\n")
    utils_helper.enforce_reproducibility(seed=cfg.muse_params.seed)
    method = cfg.mapping_method

    # return 0

    load_fmri_data = FMRIWordLevel(cfg)
    load_fmri_data.fmri_data_init()
    print("-" * 25 + "convert completed" + "-" * 25)

    if cfg.models.model_type == "LM":
        lm_emb_dict_builder = EmbedsDictsBuilder(cfg)
        lm_emb_dict_builder.build_dictionary()
        print("-" * 25 + "build bi-modal dictionaries completed!" + "-" * 25)
        lm_emb_dict_builder.process_embeddings(cfg)
        print(
            "-" * 25
            + "Extract and Decontextualize LMs representation completed!"
            + "-" * 25
        )

    if method == "procrustes":
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


if __name__ == "__main__":
    main()
