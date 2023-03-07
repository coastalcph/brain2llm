import os

import numpy as np
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
from tqdm import tqdm

from .utils.encode_util import *


device = "cuda" if torch.cuda.is_available() else "cpu"

def resnet_encode(args, encodings_path, image_classes_path):
  """
  It takes in a list of images, and returns a list of encodings
  
  Args:
    args: the command line arguments
    encodings_path: The path to the encodings file.
    image_classes_path: The path to the file containing the image classes.
  """
  if os.path.exists(encodings_path):
    print('Loading existing encodings file', encodings_path)
    encoded_image_classes = np.load(encodings_path)
    image_categories_alias = open(image_classes_path).readlines()
    image_classes_ids = [i.strip('\n').split(': ')[0] for i in image_categories_alias]
    return encoded_image_classes, image_classes_ids
  
  image_id_words_path = args.data.image_id2words
  image_dir = os.path.expanduser(args.data.image_dir)
  model_name = args.model.get('model_name', 'resnet18')

  if image_id_words_path == '':
    image_classes_ids = os.listdir(image_dir)
  else:
    ids_words = open(image_id_words_path).readlines()
    image_classes_ids = [i.strip('\n').split(': ')[0] for i in ids_words]

  def pil_image_class(image_class):
    images = []
    for filename in os.listdir(os.path.join(image_dir, image_class)):
      try:
        images.append(Image.open(os.path.join(image_dir, image_class, filename)).convert('RGB'))
      except:
        print('Failed to pil', filename)
    return images
  
  img2vec = Img2Vec(model=model_name, cuda=torch.cuda.is_available())

  def encode_one_class(images_, image_class, args):
    """
    It takes a list of images, a label, and returns a list of encoded
    images
    
    Args:
      images_: a list of images
      image_class: the class of the image (e.g. 'dog', 'cat', 'car', etc.)
      args: a dictionary of parameters
    """
    bs = 200
    batches = [images_[i:i+bs] for i in range(0, len(images_), bs)]

    features = []
    image_name = image_class
    images_names = [f"{image_name}_{i}" for i in range(len(images_))]
    with torch.no_grad():
      for batch in batches:
        features.append(img2vec.get_vec(batch, tensor=True).to('cpu').numpy())
    # print(features[0].shape)
    features = np.concatenate(features).squeeze()

    if args.model.get("need_per_object_embs", True):
      format_embeddings(
        features, 
        images_names, 
        os.path.join(args.data.per_object_embs_path,image_name))
    # print(features.shape)
    # print(np.expand_dims(features.mean(axis=0), 0).shape)
    return np.expand_dims(features.mean(axis=0), 0)

  encoded_image_classes = []

  for image_class in tqdm(image_classes_ids, mininterval=300.0, maxinterval=600.0):
    images = pil_image_class(image_class)
    encoded_image_classes.append(encode_one_class(images, image_class, args))

  encoded_image_classes = np.concatenate(encoded_image_classes)
  np.save(encodings_path, encoded_image_classes)

  with open(image_classes_path, 'w') as f:
    f.write('\n'.join(image_classes_ids))

  return encoded_image_classes, image_classes_ids
