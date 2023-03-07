import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import time as tm
import os
from .utils_b2l import remove_punct
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_bert_layer_representations(args, text_sentences_array):

    cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/transformers/models"), 
        args.model.model_name)
    model = BertModel.from_pretrained(args.model.pretrained_model,
        cache_dir=os.path.expanduser(cache_path),
        output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(args.model.pretrained_model)

    model = model.to(device)
    model.eval()

    # where to store layer-wise bert embeddings of particular length
    bert_dict = {}
    # get the token embeddings
    start_time = tm.time()

    for sent_idx, sentences in enumerate(text_sentences_array):
        # sentences_words = [i for w in sentences.lower().strip().split(' ') for (i, _) in remove_punct(w)]
        sentences_words = [remove_punct(w)[0] for w in sentences.lower().strip().split(' ')]
        sentences_words = list(filter(lambda x: x != '<punct>', sentences_words))
        bert = add_token_embedding_for_specific_word(sentences, tokenizer, model, sentences_words, bert_dict, is_avg=args.model.is_avg)

        if sent_idx % 100 == 0:
            print(f'Completed {sent_idx} out of {len(text_sentences_array)}: {tm.time() - start_time}')
            start_time = tm.time()
    
    return bert

# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
def predict_bert_embeddings(words_in_array, tokenizer, model, sentence_words, bert_dict):

    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    tokenized_text = tokenizer.tokenize(f"[CLS] {words_in_array.strip()} [SEP]")
    mask_tokenized_text = tokenized_text.copy()

    for i, word in enumerate(sentence_words):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)

        for tok in word_tokens:
            try:
                ind = mask_tokenized_text.index(tok)
            except ValueError:
                print(f"{word_tokens}")
                print(f"{tokenized_text}")
                break
            word_ind_to_token_ind[i].append(ind)
            mask_tokenized_text[ind] = "[MASK]"
        else:
            continue
        break

    indexed_tokens = tokenizer(words_in_array, return_tensors="pt")
    indexed_tokens = indexed_tokens.to(device)

    with torch.no_grad():
        outputs = model(**indexed_tokens)

    if not bert_dict:
        for layer in range(len(outputs.hidden_states)):
            bert_dict[layer] = []

    return outputs.hidden_states, word_ind_to_token_ind, tokenized_text, bert_dict

# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_bert_embedding(bert_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):

    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        bert_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))

    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.cpu().detach().numpy()
            # print(full_sequence_embedding.shape)
            bert_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word

    return bert_dict

# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: BERT tokenizer
# model: BERT model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# bert_dict: where to save the extracted embeddings
def add_token_embedding_for_specific_word(word_seq,tokenizer,model,sentences_words, bert_dict, is_avg=True):

    all_sequence_embeddings, word_ind_to_token_ind, token_text, bert_dict = predict_bert_embeddings(word_seq, tokenizer,
                                                                                         model, sentences_words, bert_dict)

    for token_inds_to_avrg in list(word_ind_to_token_ind.keys()):
        if is_avg:
            token_ind = word_ind_to_token_ind[token_inds_to_avrg]
        else:
            # only use the first token
            token_ind = [word_ind_to_token_ind[token_inds_to_avrg][0]]
            # token_ind = [token_inds_to_avrg]
        bert_dict = add_word_bert_embedding(bert_dict, all_sequence_embeddings, token_ind)

    return bert_dict

# add the embeddings for only the last word in the sequence that is not [SEP] token
def add_last_nonsep_bert_embedding(bert_dict, embeddings_to_add, specific_layer=-1):
    if specific_layer >= 0:
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        
        bert_dict[specific_layer].append(full_sequence_embedding[0,-2,:])
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            bert_dict[layer].append(full_sequence_embedding[0,-2,:])
    return bert_dict

# add the CLS token embeddings ([CLS] is the first token in each string)
def add_cls_bert_embedding(bert_dict, embeddings_to_add, specific_layer=-1):
    if specific_layer >= 0:
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        
        bert_dict[specific_layer].append(full_sequence_embedding[0,0,:])
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            bert_dict[layer].append(full_sequence_embedding[0,0,:])
    return bert_dict
