# import numba as nb
import os
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine


def get_dispersion(root_path, embeddigns_path):
    """
    Calculating the dispersion of embeddings.
    
    Args:
      root_path: the path to the folder containing the images
      embeddigns_path: the path to the embeddings file
    """
    file_path = os.path.join(root_path, embeddigns_path)
    obj_embeddings = open(file_path).readlines()
    
    name = obj_embeddings[1].rstrip().split(' ', 1)[0][:-2]

    vectors = []
    for _, line in enumerate(obj_embeddings[1:]):
        _, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        # print(vect.shape)
        vectors.append(vect)

    relations = list(combinations(vectors, 2))
    cos_results = []
    for src, tgt in relations:
        cos_dis1 = cosine(src, tgt)
        cos_results.append(cos_dis1)

    cos_avg = np.mean(cos_results)
    return cos_avg, name

def get_dispersion_multi(root_path, obj_categories):
    """
    This function takes in a root path and a list of object categories 
    to use multi-processing with some CPUs speeding up.
    
    Args:
      root_path: the path to the root directory of the dataset
      obj_categories: a list of strings, each string is the name of a category of objects
    """
    categories_dis = {}
    for obj_category in obj_categories:
        dis, name = get_dispersion(root_path, obj_category)
        categories_dis[name] = dis
    return categories_dis

