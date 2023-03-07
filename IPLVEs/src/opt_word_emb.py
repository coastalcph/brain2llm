import os

import torch
from transformers import GPT2Tokenizer, OPTModel

from .utils.encode_util import *


if not torch.cuda.is_available():
    device_ids = ["cpu"]
else:
    device_ids = [i for i in range(torch.cuda.device_count())]


def load_model(args):
    """
    It loads the model from the file.
    
    Args:
      args: a dictionary of parameters
    """
    device_map = None if len(device_ids)==1 else "auto"
    CACHE_PATH = os.path.join(
        os.path.expanduser("~/.cache/huggingface/transformers/models"), 
        args.model.model_name)

    model = OPTModel.from_pretrained(args.model.pretrained_model, \
        cache_dir=CACHE_PATH, device_map=device_map)
        
    tokenizer = GPT2Tokenizer.from_pretrained(args.model.pretrained_model)
    if not device_map:
        model = model.to(device_ids[0])
    return model, tokenizer


def text_preparation(context, tokenizer):
    """
    Preparing the input for OPT
    Takes a string argument and performs
    
    Args:
      context: the text to be tokenized
      tokenizer: a function that takes in a string and returns a list of tokens
    """
    tokenized_text = tokenizer.tokenize(f'</s>{context}')
    inputs = tokenizer(context, return_tensors="pt")
    return tokenized_text, inputs.to(device_ids[0])
