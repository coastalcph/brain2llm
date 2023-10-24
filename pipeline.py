from pathlib import Path

from src.LMs_dict_rep import EmbedsDictsBuilder
from src.biggraph_dict_rep import BigGraphEmb
from src.fmri_dataloader import FMRIWordLevel
from src.muse import Muse
from src.ridge_regression import RidgeRegression
from src.rsa import RSA
from src.utils import io_util
from src.utils import utils_helper


def main():
    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    utils_helper.enforce_reproducibility(seed=config.muse_parameters.seed)
    exp_dir = Path(f"./{config.expdir.expname}")
    method = config.method
    config = utils_helper.uniform_config(args=config)

    fmri_emb_type = "type"
    extra_dict = "poly"

    load_fmri_data = FMRIWordLevel(config)
    load_fmri_data.fmri_data_init(smoothing="Gaussian")
    print("-" * 25 + "convert completed" + "-" * 25)

    if config.model.model_type == "LM":
        lm_emb_dict_builder = EmbedsDictsBuilder(config, fmri_type=fmri_emb_type)
        lm_emb_dict_builder.build_dictionary(more_dict=extra_dict)
        print("-" * 25 + "build bi-modal dictionaries completed!" + "-" * 25)
        lm_emb_dict_builder.process_embeddings(config)
        print("-" * 25 + "Extract and Decontextualize LMs representation completed!" + "-" * 25)
    elif config.model.model_type == "VM":
        pass
    else:
        bg_dict_builder = BigGraphEmb(config)
        bg_dict_builder.extract_bg_embeddings()
        print("-" * 25 + "Extract and Biggraph representation completed!" + "-" * 25)

    if method == "Procrustes":
        procrustes_exp = Muse(config, train_eval="train", fmri_type=fmri_emb_type)
        procrustes_exp.run()
        # procrustes_exp.run(extend_exp=extra_dict)
    elif method == "rsa":
        rsa_exp = RSA(config)
        print(rsa_exp.run_rsa())
        print("-" * 25 + "RSA Done!" + "-" * 25)
    else:
        regression = RidgeRegression(config)
        regression.run_regression(fmri_type=fmri_emb_type)
        print("-" * 25 + "Regression Done!" + "-" * 25)
        eval_exp = Muse(config, train_eval="eval", fmri_type=fmri_emb_type)
        eval_exp.run()
        # eval_exp.run(extend_exp=extra_dict)

    exp_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_averaged" if config.model.is_avg else "_first"
    io_util.save_config(config,
                        exp_dir / f'exp_{config.data.dataset_name}_config{suffix}_{config.model.model_name}.yaml')


if __name__ == '__main__':
    main()

