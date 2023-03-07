import os
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Model, GPT2TokenizerFast

from .utils.encode_util import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(args):
    """
    It loads the model from the file.
    
    Args:
      args: a dictionary of parameters
    """
    CACHE_PATH = os.path.join(
        os.path.expanduser("~/.cache/huggingface/transformers/models"), 
        args.model.model_name)
    model = GPT2Model.from_pretrained(args.model.pretrained_model,
        cache_dir=os.path.expanduser(CACHE_PATH),
        output_hidden_states=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model.pretrained_model)
    return model.to(device), tokenizer


def text_preparation(context, tokenizer):
    """
    Preparing the input for GPT2
    Takes a string argument and performs
    
    Args:
      context: the text to be tokenized
      tokenizer: a function that takes in a string and returns a list of tokens
    """
    tokenized_text = tokenizer.tokenize(context)
    inputs = tokenizer(context, return_tensors="pt")

    return tokenized_text, inputs.to(device)


def average_word_embedding(contexts, model, tokenizer, args):
    """
    For each context, tokenize it, get the embedding for the required token, average the embeddings, and return
    the average embeddings for all contexts
    
    Args:
      contexts: a list of strings, each string is a sentence
      model: the model we're using to get the word embeddings
      tokenizer: a tokenizer object from the GPT2 library
      args: a dictionary of arguments
    """
    iters = tqdm(contexts, mininterval=1800.0, maxinterval=3600.0)
    words_order = [i.split(':',1)[0].lower() for i in contexts]
    
    word_current = words_order[0]
    subword_average_embedding = []
    target_word_average_embeddings = []
    words_in_sentences = [word_current]

    for ids, context in enumerate(iters):
        if word_current == words_order[ids]:
            phrase_lower = word_current.replace('_', ' ')
            context = context.split(': ',1)[1].strip('\n')

            # Find the spercific type of the phrase in the sentence
            # Find the correct index of the tokens in the sentence
            match = re.search(r"\b" + phrase_lower + r"\b", context.lower())

            # phrase_pos = context.lower().find(phrase_lower)

            tokenized_text, inputs = text_preparation(context, tokenizer)
            list_token_embeddings = get_embeddings(inputs, model)

            # The word is the first word
            if match.start() == 0:
                phrase_type = 0
                phrase = f'{context[match.start(): match.end()]}'
            # The word is after punctuation
            # elif context[phrase_pos-1] == '(' or context[phrase_pos-1] == '"':
            elif any(context[match.start()-1] == punc for punc in ('(','"','-')):
                phrase_type = 1
                phrase =  f'({context[match.start(): match.end()]}'
            # The word is not the first word
            else:
                phrase_type = 0
                phrase = f' {context[match.start(): match.end()]}'
                
            tokenized_word = tokenizer.tokenize(phrase)
            num_sub = len(tokenized_word)-1 if phrase_type else len(tokenized_word)

            # Get the first index of the word's token in the contexts
            word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[phrase_type]]
            for index_word in word_indices:
                word_embedding = list_token_embeddings[index_word: index_word + num_sub]
                text_contents = tokenized_text[index_word: index_word + num_sub]
                if tokenized_word[phrase_type:] == text_contents:
                    subword_average_embedding.append(np.mean(word_embedding, axis=0))
                    break

        if ids == len(contexts)-1:
            if args.model.get("need_per_object_embs", True):
                format_embeddings(
                    np.array(subword_average_embedding), 
                    [f"{word_current}_{i}" for i in range(len(subword_average_embedding))], \
                    os.path.join(args.data.per_object_embs_path, word_current))
            average_word_embedding = np.mean(subword_average_embedding, axis=0)
            target_word_average_embeddings.append(average_word_embedding)
            subword_average_embedding = []
            return words_in_sentences, np.array(target_word_average_embeddings)

        if words_order[ids+1] != words_order[ids]:
            if args.model.get("need_per_object_embs", True):
                format_embeddings(
                    np.array(subword_average_embedding), 
                    [f"{word_current}_{i}" for i in range(len(subword_average_embedding))], \
                    os.path.join(args.data.per_object_embs_path, word_current))
            average_word_embedding = np.mean(subword_average_embedding, axis=0)
            target_word_average_embeddings.append(average_word_embedding)
            subword_average_embedding = []
            word_current = words_order[ids+1]
            words_in_sentences.append(word_current)
