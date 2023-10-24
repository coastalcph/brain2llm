import json
from pathlib import Path

import numpy as np
import pywikibot
import torch

from .LMs_dict_rep import EmbedsDictsBuilder


class BigGraphEmb(EmbedsDictsBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.save_dir = Path(self.alias_emb_dir) / self.model_name
        self.dim = config.model.dim

    def embedding_filter(self, entity_embeddings, entity_names):
        word_list_kge = []
        word_embeddings_kge = []
        site = pywikibot.Site("en", "wikipedia")
        error_labels = set()
        for label in self.words:
            page = pywikibot.Page(site, label)
            try:
                item = pywikibot.ItemPage.fromPage(page)
            except:
                # print("No item found for " + label)
                error_labels.add(label)
            else:
                # print(item.getID())
                try:
                    index = entity_names.index("<http://www.wikidata.org/entity/" + item.getID() + ">")
                except:
                    error_labels.add(label)
                    # print("No embedding found for " + label)
                else:
                    word_list_kge.append(label)
                    word_embeddings_kge.append(entity_embeddings[index])
        return word_list_kge, torch.from_numpy(np.array(word_embeddings_kge))

    def extract_bg_embeddings(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"{self.dataset_name}_{self.model_name}_dim_{self.dim}_layer_0.pth"
        if save_path.exists():
            print(f"File {save_path} already exists. Skipping function.")
            data = torch.load(
                "/projects/nlp/data/brain/datasets/hp_fmri/experiments/biggraph_outputs/biggraph-uncased/potter_biggraph-uncased_dim_200_layer_0.pth")
            # self.build_dictionary(wordlist=data["dico"])
            # fasttext.util.download_model('en', if_exists='ignore')  # English
            # ft = fasttext.load_model('cc.en.300.bin')
            # fasttext.util.reduce_model(ft, 200)
            # save_dir = Path("./data/")
            # embeddings = torch.from_numpy(np.array([ft.get_word_vector(x) for x in data["dico"]]))
            # save_dir.mkdir(parents=True, exist_ok=True)
            # save_path = save_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_0.pth"
            # torch.save({'dico': data["dico"], 'vectors': embeddings}, save_path)
            return
        else:
            entity_embeddings = np.load(self.config.model.pretrained_model, mmap_mode='r', allow_pickle=True)
            entity_names = json.load(open(self.config.model.entity_name, "rt"))
            sub_word_list, sub_word_bg_embs = self.embedding_filter(entity_embeddings, entity_names)
            print("number of entity name:", len(sub_word_list))
            print("shape of BigGraph embeddings:", sub_word_bg_embs.size())
            self.build_dictionary(wordlist=sub_word_list)
            torch.save({'dico': sub_word_list, 'vectors': sub_word_bg_embs}, save_path)

