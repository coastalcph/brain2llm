import os

from .utils.gpt2_reps_utils import get_gpt2_layer_representations
import numpy as np

from .utils.bert_reps_utils import get_bert_layer_representations
from .utils import io_util


def save_layer_representations(model_layer_dict, model_name, save_dir):             
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_sentences_layer_{}.npy'.format(save_dir,model_name,layer),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1


def get_lm_rep(config):

    text_sentences_array = open(config.data.fmri_sentences_path).readlines()

    if 'bert' == config.model.model_alias:
        nlp_features = get_bert_layer_representations(config, text_sentences_array)
    elif 'gpt2' == config.model.model_alias:
        nlp_features = get_gpt2_layer_representations(config, text_sentences_array)
    else:
        nlp_features = []

    print(len(nlp_features), len(nlp_features[0]))

    config.model.n_layers = len(nlp_features)

    features_save_path = f"{config.data.all_text_embs_dir}/{config.model.model_name}"
    if not os.path.exists(features_save_path):
        os.makedirs(features_save_path)

    save_layer_representations(nlp_features, config.model.model_name, features_save_path)

    return len(nlp_features)


if __name__ == '__main__':

    # io_util.enforce_reproducibility(seed=42)

    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    params = io_util.load_config(args, unknown)

    get_lm_rep(config=params)
        
        
        
    
    
    

    
