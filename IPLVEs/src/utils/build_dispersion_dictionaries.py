import math
import multiprocessing as mp
import operator
import os

from .get_dispersion import *


def get_sorted_dispersion(args):
    """
    Calculate the dispersion of all embeddings of words/images and sorted them.
    save the result to a file.
    
    Args:
      args: includes the directories of embeddings and saving path.
    """
    root_path = args.data.per_object_embs_path
    # root_path = os.path.expanduser("~/Dir/datasets/dispersions/bert_words_embs")
    # root_path = os.path.expanduser("~/Dir/datasets/dispersions/gpt2_words_embs")
    # root_path = os.path.expanduser("~/Dir/datasets/dispersions/opt_words_embs")
    # root_path = os.path.expanduser("~/Dir/datasets/dispersions/seg_images_embs")
    # root_path = os.path.expanduser("~/Dir/datasets/dispersions/res_images_embs")

    categories = os.listdir(root_path)
    num_categories = len(categories)
    num_cpus = 8
    block_size = math.ceil(num_categories / num_cpus)
    p = mp.Pool(processes=num_cpus)
    
    res = [
        p.apply_async(
        func=get_dispersion_multi, 
        args=(root_path, categories[i*block_size:i*block_size+block_size])) for i in range(num_cpus)
        ]

    cos_res_list = [i.get() for i in res]
    categories_dis_dict = {}
    for i in cos_res_list:
        categories_dis_dict.update(i)
    categories_dis_sorted = sorted(categories_dis_dict.items(), key = lambda kv:(kv[1], kv[0]))
    # with open('./data/sorted_dispersion_bert.txt','w') as ssw:
    with open(args.data.setdefault("sorted_dispersion_file", f"./data/sorted_dispersion_{args.model.model_name}.txt"),'w') as ssw:
        for i in categories_dis_sorted:
            ssw.write(f"{i[0]}: {i[1]}\n")
        ssw.close()

def build_dis_dict(args, seeds):
    """
    create low, medium, high three evaluation dictionaries per seed based on the dispersion.
    
    Args:
      args: includes the sorted dispersion file, the eval dictionary path
      seeds: a list of seed words
    """
    get_sorted_dispersion(args)

    dis_path = args.data.get("sorted_dispersion_file", f"./data/sorted_dispersion_{args.model.model_name}.txt")
    dis_sorted = open(dis_path).readlines()
    dis_sorted_info = [i.split(': ', 1)[0] for i in dis_sorted]

    # dict_path = './data/dictionaries_id2w'
    dict_path = args.data.get("origin_dict_dir", "./data/origin_dict")
    dict_list = os.listdir(dict_path)
    dis_dict_path = f'./data/dictionaries_{args.model.model_type}_dispersion'
    if not os.path.exists(dis_dict_path):
        os.makedirs(dis_dict_path)
    # seeds = [203, 255, 633, 813, 881]
    for seed in seeds:
        for dico in dict_list:
            if "eval" in dico and str(seed) in dico:
                dictionaries = open(os.path.join(dict_path, dico)).readlines()
                eval_ids_origin = [i.split(' ', 1)[0] for i in dictionaries]
                eval_words = [i.strip().split(' ', 1)[1] for i in dictionaries]
                if args.model.model_type == 'VM':
                    sorted_index = [dis_sorted_info.index(i) for i in eval_ids_origin]
                else:
                    sorted_index = [dis_sorted_info.index(i) for i in eval_words]
                eval_zipped = list(zip(eval_ids_origin, eval_words, sorted_index))
                sorted_res = sorted(eval_zipped, key=operator.itemgetter(2))
                # eval_zipped.sort(key=dis_sorted_ids.index)
                block_size = math.ceil(len(sorted_res) / 3)
                id_final, word_final, _ = zip(*sorted_res)
                # id_final = list(id_final)
                # word_final = list(word_final)
                final_list = list(zip(id_final, word_final))
                # print(final_list[0:2])
                for block_idx, block_name in enumerate(['low', 'medium', 'high']):
                    with open(os.path.join(dis_dict_path, f'eval_{args.model.model_name}_{seed}_{block_name}.txt'), 'w') as wd:
                        for (id_s, word) in final_list[block_idx*block_size : block_size * (block_idx + 1)]:
                            wd.write(f'{id_s} {word}\n')
                        wd.close()
                        # wd.write('\n'.join(sorted_res))

# if __name__ == "__main__":
#     # model = 'bert'
#     # model_type = 'language'

    
    
                


