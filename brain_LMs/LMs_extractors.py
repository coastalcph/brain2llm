import os
from pathlib import Path

import fasttext
import fasttext.util
import numpy as np
import openai
import torch
from openai.embeddings_utils import get_embedding
from sklearn.decomposition import PCA

from .utils.LM_reps_utils import LMEmbedding


class LMExtractor:
    def __init__(self, config, words=None):
        self.config = config
        self.words = words
        self.dataset_name = config.data.dataset_name
        self.model_name = config.model.model_name
        self.model_alias = config.model.model_alias
        self.num_tr = config.data.tr_num
        self.alias_emb_dir = Path(config.data.alias_emb_dir)
        self.suffix = "averaged" if config.model.is_avg else "first"
        self.save_dir = self.alias_emb_dir / self.suffix / self.model_name
        self.fmri_sentences_path = Path(config.data.fmri_sentences_path)
        self.seed = config.muse_parameters.seed
        self.model_dim = config.model.dim

    def get_lm_rep(self):
        with open(self.fmri_sentences_path, 'r') as sent_reader:
            text_sentences_array = sent_reader.readlines()
            sent_reader.close()
        if self.model_alias == "ft":
            self.fasttext_emb()
            return
        elif self.model_alias in ["bert", "gpt2", "opt"]:
            embeddings_extractor = LMEmbedding(self.config, text_sentences_array)
            all_context_words, nlp_features_dict = embeddings_extractor.get_lm_layer_representations()
        elif self.model_alias == "openai":
            self.openai_api_emb()
            return
        else:
            nlp_features_dict = {}

        print(len(nlp_features_dict), len(nlp_features_dict[0]))
        features_save_path = self.save_dir
        features_save_path.mkdir(parents=True, exist_ok=True)
        self.save_layer_representations(all_context_words, nlp_features_dict)

    def fasttext_emb(self):
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        fasttext.util.download_model('en', if_exists='ignore')  # English
        ft = fasttext.load_model('cc.en.300.bin')
        embeddings = torch.from_numpy(np.array([ft.get_word_vector(x) for x in self.words]))
        save_path = save_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_0.pth"
        torch.save({'dico': self.words, 'vectors': embeddings}, save_path)
        if self.num_tr < self.model_dim:
            fasttext.util.reduce_model(ft, self.num_tr)
            embeddings = torch.from_numpy(np.array([ft.get_word_vector(x) for x in self.words]))
            save_path = save_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_0.pth"
            torch.save({'dico': self.words, 'vectors': embeddings}, save_path)

    def openai_api_emb(self):

        openai.api_key = os.getenv("OPENAI_API_KEY")

        embedding_model = "text-embedding-ada-002"
        # embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        save_dir = self.save_dir
        # encoding = tiktoken.get_encoding(embedding_encoding)
        embeddings = np.array([get_embedding(x, engine=embedding_model) for x in self.words])
        save_dir.mkdir(parents=True, exist_ok=True)
        if embeddings.shape[1] > self.num_tr:
            pca = PCA(n_components=self.num_tr, random_state=self.seed)
            reduced_embeddings = pca.fit_transform(embeddings)
            save_path = save_dir / f"{self.dataset_name}_{self.model_name}_dim_{reduced_embeddings.shape[1]}_layer_0.pth"
            torch.save({'dico': self.words, 'vectors': torch.from_numpy(reduced_embeddings)}, save_path)
        save_path = save_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_0.pth"
        torch.save({'dico': self.words, 'vectors': torch.from_numpy(embeddings)}, save_path)

    def save_layer_representations(self, words, model_layer_dict):
        for layer in model_layer_dict.keys():
            embeddings = np.vstack(model_layer_dict[layer])
            if embeddings.shape[1] > self.num_tr:
                pca = PCA(n_components=self.num_tr, random_state=self.seed)
                reduced_embeddings = pca.fit_transform(embeddings)
                torch.save({"dico": words, "vectors": torch.from_numpy(reduced_embeddings).float()},
                           str(self.save_dir / f"{self.dataset_name}_{self.model_name}_dim_{self.num_tr}_length_sentences_layer_{layer}.pth")
                           )
            torch.save({"dico": words, "vectors": torch.from_numpy(embeddings).float()},
                       str(self.save_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_length_sentences_layer_{layer}.pth")
                       )
        print(f"Saved extracted features to {str(self.save_dir)}")
