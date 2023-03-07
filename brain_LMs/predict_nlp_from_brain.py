import argparse
import numpy as np
import os
from utils import io_util
from utils.utils_b2l import run_class_time_CV_fmri_crossval_ridge
import torch
import random


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    seed = 42
    enforce_reproducibility()

    parser = io_util.create_args_parser()
    config, unknown = parser.parse_known_args()
    args = io_util.load_config(config, unknown)

    predict_feat_dict = {'nlp_feat_type': args.data.nlp_feat_type,
                         'nlp_feat_dir': args.data.nlp_feat_dir,
                         'layer': args.data.layer,
                         'seq_len': args.data.sequence_length,
                         'time_fmri_path': args.data.time_fmri_path,
                         'runs_fmri_path': args.data.runs_fmri_path,
                         'time_words_path': args.data.time_words_path,
                         'subject': args.data.subject,
                         'output_dir': args.data.output_dir}

    # loading fMRI data

    data = np.load(f'/home/kfb818/projects/b2le/debugs_data/fmri_word_level/potter-sub--{args.data.subject}-2.0-2.0-0.txt')
    corrs_t, _, _, _, _, test_d = run_class_time_CV_fmri_crossval_ridge(data,
                                                                        predict_feat_dict)

    if not os.path.exists(args.data.output_dir):
        os.makedirs(args.data.output_dir)
    fname = 'test_brain_data_{}_with_{}_layer_{}_len_{}'.format(args.data.subject, args.data.nlp_feat_type,
                                                                args.data.layer, args.data.sequence_length)
    print('saving: {}'.format(args.data.output_dir + fname))

    np.save(args.data.output_dir + fname + '.npy', {'test_d': test_d})
