import torch
import numpy as np
from transformers import GPT2Model, GPT2TokenizerFast
import time as tm
import os
from .utils_b2l import remove_punct
import re


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_gpt2_layer_representations(args, text_sentences_array):
    cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/transformers/models"),
        args.model.model_name)
    model = GPT2Model.from_pretrained(args.model.pretrained_model,
                                      cache_dir=os.path.expanduser(cache_path),
                                      output_hidden_states=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model.pretrained_model)

    model = model.to(device)
    model.eval()

    # where to store layer-wise bert embeddings of particular length
    gpt2_dict = {}
    # get the token embeddings
    start_time = tm.time()

    for sent_idx, sentences in enumerate(text_sentences_array):
        # sentences_words = [i for w in sentences.lower().strip().split(' ') for (i, _) in remove_punct(w)]
        sentences_words = [remove_punct(w)[0] for w in sentences.lower().strip().split(' ')]
        sentences_words = list(filter(lambda x: x != '<punct>', sentences_words))
        gpt2 = add_token_embedding_for_specific_word(sentences, tokenizer, model, sentences_words, gpt2_dict,
                                                     is_avg=args.model.is_avg)

        if sent_idx % 5 == 0:
            print(f'Completed {sent_idx} out of {len(text_sentences_array)}: {tm.time() - start_time}')
            start_time = tm.time()

    return gpt2


# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
def predict_gpt2_embeddings(words_in_array, tokenizer, model, sentence_words, gpt2_dict):
    word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    words_in_array = words_in_array.lower().strip()
    words_mask = list(words_in_array)
    tokenized_text = tokenizer.tokenize(words_in_array)
    mask_tokenized_text = tokenized_text.copy()

    for i, word in enumerate(sentence_words):
        word_ind_to_token_ind[i] = []  # initialize token indices array for current word

        match = re.search(r"\b" + word.lower() + r"\b", ''.join(words_mask))
        if not match:
            print(match)
            print(word)
            print(words_in_array)
        if match.start() == 0:
            word_tokens = tokenizer.tokenize(word)
            # words_mask[match.start():match.end()] = " " * (match.end() - match.start())
        else:
            if words_in_array[match.start()-1] in  ('(', '\"', '-', '\'', "‘"):
                word_tokens = tokenizer.tokenize(word)
                # words_mask[match.start():match.end()] = " " * (match.end() - match.start())
            else:
                word_tokens = tokenizer.tokenize(f" {word}")
        words_mask[match.start():match.end()] = " " * (match.end() - match.start())

        for tok in word_tokens:
            try:
                ind = mask_tokenized_text.index(tok)
            except ValueError:
                print(i, word, words_in_array, word_tokens)
                print(tokenized_text)
                print()
            word_ind_to_token_ind[i].append(ind)
            mask_tokenized_text[ind] = "[MASK]"
            # n_seq_tokens = n_seq_tokens + 1

    indexed_tokens = tokenizer(words_in_array, return_tensors="pt")
    indexed_tokens = indexed_tokens.to(device)

    with torch.no_grad():
        outputs = model(**indexed_tokens)

    if not gpt2_dict:
        for layer in range(len(outputs.hidden_states)):
            gpt2_dict[layer] = []

    return outputs.hidden_states, word_ind_to_token_ind, tokenized_text, gpt2_dict


# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_gpt2_embedding(gpt2_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):

    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        gpt2_dict[specific_layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.cpu().detach().numpy()
            # print(full_sequence_embedding.shape)
            gpt2_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :],
                                            0))  # avrg over all tokens for specified word

    return gpt2_dict


# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: gpt2 tokenizer
# model: gpt2 model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# gpt2_dict: where to save the extracted embeddings
def add_token_embedding_for_specific_word(word_seq, tokenizer, model, sentences_words, gpt2_dict, is_avg=True):
    all_sequence_embeddings, word_ind_to_token_ind, token_text, gpt2_dict = predict_gpt2_embeddings(word_seq, tokenizer,
                                                                                                    model,
                                                                                                    sentences_words,
                                                                                                    gpt2_dict)

    for token_inds_to_avrg in list(word_ind_to_token_ind.keys()):
        if is_avg:
            token_ind = word_ind_to_token_ind[token_inds_to_avrg]
        else:
            # only use the first token
            token_ind = [word_ind_to_token_ind[token_inds_to_avrg][0]]
            # token_ind = [token_inds_to_avrg]
        gpt2_dict = add_word_gpt2_embedding(gpt2_dict, all_sequence_embeddings, token_ind)

    return gpt2_dict

