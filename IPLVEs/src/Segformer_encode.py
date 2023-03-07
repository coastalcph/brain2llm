import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils.encode_util import *


# This class inherits from the Dataset class and overrides the __len__ and __getitem__ methods
class ImageDataset(Dataset):
    def __init__(self, image_dir, image_category_id, extractor, resolution) -> None:
        super(ImageDataset, self).__init__()
        self.image_root = image_dir
        self.id_name_pairs = open(image_category_id).readlines()
        self.ids = [i.strip('\n').split(': ')[0] for i in self.id_name_pairs]
        self.names = [i.strip('\n').split(': ')[1] for i in self.id_name_pairs]
        self.extractor = extractor
        self.MAX_SIZE = 200
        self.RESOLUTION_HEIGHT = resolution
        self.RESOLUTION_WIDTH = resolution
        self.CHANNELS = 3

    def __len__(self):
        return len(self.id_name_pairs)
    
    def __getitem__(self, index):
        images = []
        category_path = os.path.join(self.image_root, self.ids[index])
        for filename in os.listdir(category_path):
            try:
                images.append(Image.open(os.path.join(category_path, filename)).convert('RGB'))
            except:
                print('Failed to pil', filename)

        category_size = len(images)
        values = self.extractor(images=images, return_tensors="pt")
        inputs = torch.zeros(self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH)
        with torch.no_grad():
            inputs[:category_size,:,:,:].copy_(values.pixel_values)

        return inputs, (self.ids[index], category_size)
        
device = "cuda" if torch.cuda.is_available() else "cpu"

def segformer_encode(model, dataloader, encoding_path, image_classes_path, args):
    """
    Use segformer to get image embeddings.
    
    Args:
      model: the model to use for encoding
      dataloader: a dataloader object that contains the data to be encoded
      encoding_path: the path to save the encoding
      image_classes_path: path to the file containing the image classes
      args: the arguments passed to the script
    """
    if os.path.exists(encoding_path):
        print('Loading existing encoding file', encoding_path)
        encoded_image_classes = np.load(encoding_path)
        image_categories = open(image_classes_path).read().strip().lower().replace(' ','_').split('\n')
        return encoded_image_classes, image_categories
    
    model.eval()
    model = model.to(device)

    categories_encode = []
    image_categories = []

    images_name = []
    for inputs, (names, category_size) in tqdm(dataloader, mininterval=1800.0, maxinterval=3600.0):
        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1, inputs_shape[2],inputs_shape[3],inputs_shape[4]).to(device)
        
        if 'b5' in args.model.model_name:
            # for b5
            bs_features = []
            bs = 100
            batches = [inputs[i:i+bs] for i in range(0, len(inputs), bs)]
            for i in batches:
                with torch.no_grad():
                    outputs = model(pixel_values=i)
                bs_features.append(outputs.last_hidden_state)
            bs_features = torch.cat(bs_features, axis=0).cpu()
            chunks = torch.chunk(bs_features, inputs_shape[0], dim=0)
        else:
            # for b0-4
            with torch.no_grad():
                outputs = model(pixel_values=inputs)
            chunks = torch.chunk(outputs.last_hidden_state.cpu(), inputs_shape[0], dim=0)

        for idx, chip in enumerate(chunks):
            # features for every image
            images_features = np.mean(chip[:category_size[idx]].numpy(), axis=(2,3), keepdims=True).squeeze()
            # features for categories
            category_feature = np.expand_dims(images_features.mean(axis=0), 0)

            image_categories.append(names[idx])
            images_name = [f"{names[idx]}_{i}" for i in range(category_size[idx])]

            categories_encode.append(category_feature)

            if args.model.get("need_per_object_embs", True):
                # save images' features
                format_embeddings(
                    images_features, 
                    images_name, 
                    os.path.join(args.data.per_object_embs_path, names[idx]))

    categories_encode = np.concatenate(categories_encode)

    np.save(encoding_path, categories_encode)

    with open(image_classes_path, 'w') as f:
        f.write('\n'.join(image_categories))
    
    return categories_encode, image_categories
