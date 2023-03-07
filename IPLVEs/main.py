import os

import numpy as np
from transformers import SegformerFeatureExtractor, SegformerModel

from .src import average_word_embedding, load_model
from .src.resnet_encode_categories import resnet_encode
from .src.Segformer_encode import *
from .src.utils import encode_util, io_util
from .src.utils.build_dico_shuffle import *
from .src.utils.build_dispersion_dictionaries import *
from .src.utils.get_frequency import *
from .src.utils.get_polysemy import *


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


def shuffle_dictionary(args, seeds):
    """
    It takes a dictionary of arguments and a list of seeds, and returns a list of dictionaries of
    arguments, where each dictionary is a shuffled version of the original dictionary
    
    Args:
      args: a dictionary of arguments
      seeds: a list of seeds to use for shuffling
    """
    ids_words_path = args.data.image_id2words
    dictionary_dir = args.data.setdefault("origin_dict_dir", "./data/origin_dict")

    if not os.path.exists(dictionary_dir):
        os.mkdir(dictionary_dir)
    
    for seed in seeds:
        train_dico, eval_dico, test_dico = dico_build(ids_words_path, seed)
        dico_write(write_dir=dictionary_dir, dicos=[train_dico, eval_dico, test_dico], seed=seed)

def build_image_classes_maps(args):
    """
    build a dictionary of image ids and their corresponding classes' names which is in the wordlist.
    
    Args:
      args: config parameters
    """
    wordlist_path = args.data.get("wordlist_path", "./data/ordered_wordlist.txt")
    image_id_maps_path = args.data.get("image_maps",'./data/imagenet21k_ids_names.txt')
    
    words = open(wordlist_path).read().strip().split('\n')
    image_classes_ids = os.listdir(os.path.expanduser(args.data.image_dir))
    image_maps = open(image_id_maps_path).read().strip().split('\n')
    image_dict = {}
    for image_map in image_maps:
        image_dict[image_map.split(': ')[0]] = image_map.split(': ')[1].split(', ')

    count = 0
    # image_with_sentences = []
    image_ids_using = []
    image_words_using = []
    for id in image_classes_ids:
        curr = list(filter(lambda x: x in words, image_dict[id]))
        if curr:
            count += 1
            for w in curr:
                image_ids_using.append(id)
                image_words_using.append(w)
            # image_with_sentences.extend(curr)
    print(count)

    with open(args.data.image_id2words, 'w+') as ids_w:
        for x,y in zip(image_ids_using, image_words_using):
            ids_w.write(f'{x}: {y}\n')
        ids_w.close()


def main_function(args, logger):
    """
    Encode languages or images
    
    Args:
      args: a list of arguments passed to the script.
    """

    model_type = args.model.get('model_type', 'LM')

    encodings_path = os.path.join(args.data.output_dir, f"{args.model.model_name}_encodings.npy")
    embeddings_path = os.path.join(args.data.emb_dir, args.model.model_name)

    if not os.path.exists(args.data.output_dir):
        os.makedirs(args.data.output_dir)
    if not os.path.exists(args.data.emb_dir):
        os.makedirs(args.data.emb_dir)
    if not os.path.exists(args.data.per_object_embs_path) and args.model.need_per_object_embs:
        os.makedirs(args.data.per_object_embs_path)

    if model_type == "LM":
        _, sentences = encode_util.load_texts(args)

        if not os.path.exists(encodings_path):
            # print(words, sentences)
            model, tokenizer = load_model(args)
            logger.info('==========Wording embedding==========')
            words_in_sentences, targets = average_word_embedding(sentences, model, tokenizer, args)
            logger.info(f'target shape: {targets.shape}')
            with open(args.data.ordered_wordlist_path,'w') as word_w:
                for word in words_in_sentences:
                    word_w.write(f'{word}\n')
                word_w.close()
            np.save(encodings_path, targets)
            data_list = words_in_sentences
        else:
            logger.info('Encodings file exists!')
            targets = np.load(encodings_path)
            words_in_sentences = open(args.data.ordered_wordlist_path).read().strip().lower().split('\n')
            data_list = words_in_sentences
    elif model_type == "VM":
        if not os.path.exists(args.data.image_id2words):
            build_image_classes_maps(args)
        image_class_list_path = os.path.join(args.data.output_dir, f"{args.model.model_name}_image_classes.txt")
        
        if args.model.get('model_alias', 'resnet') == "resnet":
            # targets, image_classes = resnet_encode(
            #     args.model.get('model_name', 'resnet18'),
            #     os.path.expanduser(args.data.image_dir), 
            #     encodings_path, 
            #     image_class_list_path, 
            #     args.data.image_id2words)
            targets, image_classes = resnet_encode(
                args,
                encodings_path, 
                image_class_list_path)
            data_list = image_classes
        elif args.model.get('model_alias', 'segformer') == "segformer":
            CACHE_PATH = os.path.join(
                os.path.expanduser("~/.cache/huggingface/transformers/models"), 
                args.model.model_name)
            feature_extractor = SegformerFeatureExtractor.from_pretrained(
                args.model.pretrained_model,
                cache_dir=CACHE_PATH,
                output_hidden_states=True)
            model = SegformerModel.from_pretrained(
                args.model.pretrained_model,
                cache_dir=CACHE_PATH,
                output_hidden_states=True, 
                return_dict=True)
            
            Resolution = int(args.model.model_name[-3:])
            imageset = ImageDataset(
                image_dir=os.path.expanduser(args.data.image_dir), 
                image_category_id=args.data.image_id2words, 
                extractor=feature_extractor, 
                resolution=Resolution)
            batch_size = 1
            image_dataloader = torch.utils.data.DataLoader(
                imageset, 
                batch_size=batch_size, 
                num_workers=8, 
                pin_memory=True)
            targets, image_categories = segformer_encode(
                model, 
                dataloader=image_dataloader, 
                encoding_path=encodings_path, 
                image_classes_path=image_class_list_path,
                args=args)
            data_list = image_categories
        else:
            NotImplementedError
        logger.info('==========Embedding complete==========')
    logger.info('==========Format embeddings==========')

    if args.model.n_components < targets.shape[1]:
        logger.info('==========Reducing dimensionality==========')
        output_reduced_dir = args.data.get('reduced_output_dir', f'./data/outputs/{args.model.model_type}_out_reduced')
        embeddings_reduced_dir = args.data.get('reduced_emb_dir', f'./data/embeddings/{args.model.model_type}_emb_reduced')
        if not os.path.exists(output_reduced_dir):
            os.makedirs(output_reduced_dir)
        if not os.path.exists(embeddings_reduced_dir):
            os.makedirs(embeddings_reduced_dir)

        reduced_encodings_path = os.path.join(
            output_reduced_dir, 
            f'{args.model.model_name}_encodings_reduced_{args.model.n_components}.npy')
        embeddings_path = os.path.join(
            embeddings_reduced_dir, 
            f"{args.model.model_name}_reduced_{args.model.n_components}")

        final_target = encode_util.reduce_encoding_size(
            targets, 
            reduced_encodings_path, 
            args.model.n_components)
        logger.info('==========Reducing dimensionality is completed!==========')
    else:
        final_target = targets
    
    encode_util.format_embeddings(final_target, data_list, embeddings_path)

    logger.info('==========Format complete==========')


if __name__ == "__main__":
    seed = 42
    enforce_reproducibility()

    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    exp_dir = config.expdir.expname

    logger = encode_util.initialize_exp(config)

    main_function(config, logger)

    seeds = np.random.randint(0, 1000, size=5)

    # build dictionary image to word
    if config.model.model_type == "VM":
        shuffle_dictionary(config, seeds)

    # For dispersion, polysemy, frequency experiments
    build_dis_dict(config, seeds)
    if config.model.model_type == "VM":
        build_polyseme_dict(get_polyseme(config), config, seeds)
        build_frequency_dict(get_frequency_rank(config), config, seeds)
    
    io_util.save_config(config, os.path.join(exp_dir, 'config.yaml'))
