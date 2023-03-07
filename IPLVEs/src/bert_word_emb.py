import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from .utils.encode_util import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(args):
    CACHE_PATH = os.path.join(
        os.path.expanduser("~/.cache/huggingface/transformers/models"), 
        args.model.model_name)
    model = BertModel.from_pretrained(args.model.pretrained_model,
        cache_dir=os.path.expanduser(CACHE_PATH),
        output_hidden_states=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model.pretrained_model)
    return model.to(device), tokenizer

def text_preparation(context, tokenizer):
    """
    Preparing the input for BERT
    Takes a string argument and performs

    Args:
      context: the text need to tokenized
      tokenizer: BERT tokenizer.
    """
    marked_text = f"[CLS] {context} [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    inputs = tokenizer(context, return_tensors="pt")
    return tokenized_text, inputs.to(device)

def average_word_embedding(contexts, model, tokenizer, args):
    """
    For BERT.
    For each word in the list of words, getting the average embedding of the word in the context
    
    Args:
      contexts: list of contexts
      model: the BERT model you want to use.
      tokenizer: The tokenizer that we used to train the model.
    
    Returns:
      The average embedding of the word in the context.
    """
    iters = tqdm(contexts, mininterval=300.0, maxinterval=600.0)
    words_order = [i.split(':',1)[0].lower() for i in contexts] # size 106564
    
    word_current = words_order[0]
    subword_average_embedding = []
    target_word_average_embeddings = []
    words_in_sentences = [word_current]

    for ids, context in enumerate(iters):
        if word_current == words_order[ids]:
            phrase = word_current.replace('_', ' ')
            tokenized_word = tokenizer.tokenize(phrase)
            num_sub = len(tokenized_word)
            context = context.split(': ',1)[1].strip('\n')
            
            tokenized_text, inputs = text_preparation(context.strip('\n'), tokenizer)
            list_token_embeddings = get_embeddings(inputs, model)

            # Get the first indices of the word's token in the contexts
            word_indices = [i for i,x in enumerate(tokenized_text) if x == tokenized_word[0]]
            for index_word in word_indices:
                word_embedding = list_token_embeddings[index_word: index_word + num_sub]
                if tokenized_word == tokenized_text[index_word: index_word + num_sub]:
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
