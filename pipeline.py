from src.utils import utils_helper, io_util
import os
from brain_LMs.extract_LMs_features import get_lm_rep
from src.convert_frmi_wordlevel import convert
from src.build_dict_rep import *
from IPLVEs.MUSE import supervised_wordlevel
import wandb
import copy
import fasttext
import fasttext.util


def run_lms(config, num_folds, model_name, layers, dataset_name, take_tokens):
    project_name = f"b2l_procrustes_{config.model.model_alias}_{take_tokens}"
    for fold in range(num_folds):
        for layer in range(config.model.get("n_layers", layers)):
            config.muse_parameters.tgt_lang = f"{model_name}.{layer}"
            for sub in range(1, config.data.num_subjects+1):
                config.muse_parameters.src_lang = f"brain_{sub}"

                wandb.init(project=project_name,
                         name=f"{config.muse_parameters.src_lang}_{config.muse_parameters.tgt_lang}_{fold}",
                         tags =[f"{config.muse_parameters.src_lang}",f"{config.muse_parameters.tgt_lang}".split('.')[0],f"layer_{layer}", f"fold_{fold}"],
                         group=f"{config.muse_parameters.tgt_lang}")

                muse_params = copy.deepcopy(config.muse_parameters)
                muse_params.dico_train = f"{config.muse_parameters.dico_train}/decon_train_fold_{fold}.txt"
                muse_params.dico_eval = f"{config.muse_parameters.dico_eval}/decon_test_fold_{fold}.txt"
                muse_params.src_emb = f"{config.muse_parameters.src_emb}/{dataset_name}-sub--{sub}-{config.convert_parameters.lookback}-{config.convert_parameters.lookout}-{config.convert_parameters.vec_dim}.txt"
                muse_params.tgt_emb = f"{config.muse_parameters.tgt_emb}_{take_tokens}/{model_name}/{model_name}_{layer}.txt"
                supervised_wordlevel.muse_supervised(muse_params)
                wandb.finish()

def run_ft(config, num_folds, model_name, dataset_name):
    for fold in range(num_folds):
        config.muse_parameters.tgt_lang = f"{model_name}"
        for sub in range(1, config.data.num_subjects+1):
            config.muse_parameters.src_lang = f"brain_{sub}"
            wandb.init(project=f"b2ft_{dataset_name}",
                             name=f"{config.muse_parameters.src_lang}_{config.muse_parameters.tgt_lang}_{fold}",
                             tags =[f"{config.muse_parameters.src_lang}",f"{config.muse_parameters.tgt_lang}".split('.')[0], f"fold_{fold}"],
                             group=f"{config.muse_parameters.tgt_lang}")

            muse_params = copy.deepcopy(config.muse_parameters)
            muse_params.dico_train = f"{config.muse_parameters.dico_train}/decon_train_fold_{fold}.txt"
            muse_params.dico_eval = f"{config.muse_parameters.dico_eval}/decon_test_fold_{fold}.txt"
            muse_params.src_emb = f"{config.muse_parameters.src_emb}/{dataset_name}-sub--{sub}-{config.convert_parameters.lookback}-{config.convert_parameters.lookout}-{config.convert_parameters.vec_dim}.txt"
            muse_params.tgt_emb = f"{config.muse_parameters.tgt_emb}/{model_name}_{dataset_name}_{config.muse_parameters.emb_dim}.txt"
            supervised_wordlevel.muse_supervised(muse_params)
            wandb.finish()

def fasttext_emb(words, dataname, config):
    n = 300
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    # embeddings = [ft.get_sentence_vector(x.replace('-', ' ')) for x in wl]
    embeddings = [ft.get_word_vector(x) for x in words]
    with open(f'{config.data.all_text_embs_dir}/fasttext_{dataname}_{n}.txt', 'w+') as fe:
        fe.write(f'{len(words)} {n}\n')
        for index, embed in enumerate(embeddings):
            embeds = ' '.join([str(i) for i in embed])
            fe.write(f'{words[index]} {embeds}\n')
        fe.close()


def main():

    utils_helper.enforce_reproducibility(seed=42)

    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    exp_dir = config.expdir.expname
    model_name = config.model.model_name
    num_folds = config.data.num_folds
    dataset_name = config.data.dataset_name
    if config.model.is_avg:
        take_token = "averaged"
    else:
        take_token = "first"

    # Process fmri data
    convert(config.data.fmri_dir,config.data.word_level_fmri_rep_dir, dataset_name, config.convert_parameters.vec_dim,
            config.data.num_subjects, config.convert_parameters.normalize, config.convert_parameters.lookout,
            config.convert_parameters.lookback, config.convert_parameters.delay, config.convert_parameters.smoothing
            )
    print("=============convert completed!=============")

    # Build bi-modal dictionaries
    fmri_words = open(
        f"{config.data.word_level_fmri_rep_dir}/{dataset_name}-sub--{config.data.num_subjects}-{config.convert_parameters.lookback}-{config.convert_parameters.lookout}-{config.convert_parameters.vec_dim}.txt").readlines()
    wordlist = [i.split(' ', 1)[0] for i in fmri_words[1:]]
    words = list(dict.fromkeys([i.split('_', 1)[0] for i in wordlist]))
    build_dictionaries(wordlist, config.data.dict_dir, num_folds)
    print("=============build bi-modal dictionaries completed!=============")

    # Extract LMs representations
    layers = get_lm_rep(config)
    print("=============Extract LMs representation completed!=============")

    # Process (decontextualization) LMs words embeddings file
    for layer in range(config.model.get("n_layers", layers)):
        lm_embeddings = np.load(
            f"{config.data.all_text_embs_dir}/{model_name}/{model_name}_length_sentences_layer_{layer}.npy",
            allow_pickle=True)
        process_lm_word_embeddings(model_name, lm_embeddings, wordlist, layer, config.data.word_decontextualized_embs_dir, config.model.is_avg)
    print("=============decontextualization LMs representation completed!=============")

    # # Process static LMs
    # fasttext_emb(words, dataname=dataset_name, config=config)
    # print("=============Extract static word representation completed!=============")
    # # Procruste Analysis Baseline (fasttext)
    # run_ft(config, num_folds, model_name, dataset_name)

    # Procruste Analysis
    run_lms(config, num_folds, model_name, config.model.n_layers, dataset_name, take_token)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    io_util.save_config(config, os.path.join(exp_dir, f'{take_token}_{model_name}_config.yaml'))

if __name__ == '__main__':
    main()
