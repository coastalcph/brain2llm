import bisect
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection


class FMRIWordLevel:
    def __init__(self, config):
        self.data_dir = Path(config.data.fmri_dir)
        self.outfile_dir = Path(config.data.word_level_fmri_rep_dir)
        self.dataset_name = config.data.dataset_name
        # self.vec_dim = int(min(config.data.tr_num, config.convert_parameters.vec_dim))
        self.vec_dim = config.convert_parameters.vec_dim
        self.subjects = config.data.num_subjects
        # self.normalize = config.convert_parameters.normalize
        self.lookout = config.convert_parameters.lookout
        self.lookback = config.convert_parameters.lookback
        self.delay = config.convert_parameters.delay
        self.seed = config.muse_parameters.seed
        # self.smoothing = config.convert_parameters.smoothing

    def check_files_exist(self, vec_dim):
        files_exist = all(
            (
                        self.outfile_dir / f"{self.dataset_name}-{t}-sub--{sub}-{self.lookback}-{self.lookout}-{vec_dim}.pth").exists()
            for sub in range(1, self.subjects + 1) for t in ["type", "token"]
        )
        return files_exist

    def fmri_data_init(self, smoothing):
        original_size_files_exist = self.check_files_exist(vec_dim=0)
        dim_size_files_exist = self.check_files_exist(vec_dim=self.vec_dim)
        if dim_size_files_exist:
            print(f"Word level fMRI dim size {self.vec_dim} already exists. Skipping dictionary building.")
        else:
            self.convert_fmri(smoothing)
        if original_size_files_exist:
            print("Word level fMRI original size already exists. Skipping dictionary building.")
        else:
            self.vec_dim = 0
            self.convert_fmri(smoothing)

    def convert_fmri(self, smoothing):
        if smoothing == "Gaussian":
            if self.dataset_name == "potter":
                words, times, fmri_timestamp, subjects_dims, vecs = self.hp_dataloader()
            else:
                words, times, fmri_timestamp, subjects_dims, vecs = self.nat_dataloader()
            vecs = self.map_words2vecs(words, times, fmri_timestamp, vecs, self.lookout,
                                       self.lookback, self.delay, normalize=False)
        else:
            words, subjects_dims, vecs = self.pereira_dataloader()
        self.save_files(words, torch.from_numpy(vecs), subjects_dims)

    def pereira_dataloader(self):
        # Here we change the name of original dirs for easy implementation (01-->1, P01-->M12)
        self.outfile_dir.mkdir(parents=True, exist_ok=True)
        # pca = PCA(n_components=self.vec_dim, random_state=self.seed) if self.vec_dim else None
        pca = GaussianRandomProjection(random_state=self.seed, n_components=self.vec_dim) if self.vec_dim else None
        words = []
        reduced_vecs = None
        subjects_dims = [0]
        remove_idx = None
        for i in range(1, self.subjects + 1):
            print("Extracting vectors for subject {}".format(i))
            matfile = self.data_dir / f"subject_{i}.mat"
            data = sio.loadmat(str(matfile))
            if i == 1:
                for idx, curr_word in enumerate(data["keyConcept"]):
                    if curr_word != "counting":
                        words.append(curr_word[0][0])
                    else:
                        remove_idx = idx
                        print("REMOVE counting idx:", remove_idx)
                num_words, num_voxels = data["examples"].shape
                reduced_vecs = np.full((num_words, num_voxels * self.subjects * 2), np.nan)
            vecs = data["examples"]
            if pca is not None:
                reduced_vecs[:, (i - 1) * self.vec_dim: i * self.vec_dim] = pca.fit_transform(vecs)
            else:
                subjects_dims.append(subjects_dims[i - 1] + vecs.shape[1])
                reduced_vecs[:, subjects_dims[i - 1]: subjects_dims[i]] = vecs
        reduced_vecs = reduced_vecs[:, ~np.isnan(reduced_vecs).all(axis=0)]
        if remove_idx is not None:
            reduced_vecs = np.delete(reduced_vecs, remove_idx, 0)
        return words, subjects_dims, reduced_vecs

    def nat_dataloader(self):
        self.outfile_dir.mkdir(parents=True, exist_ok=True)
        pca = GaussianRandomProjection(n_components=self.vec_dim, random_state=self.seed) if self.vec_dim else None
        words = []
        times = []
        reduced_vecs = None
        subjects_dims = [0]
        for i in range(1, self.subjects + 1):
            print("Extracting vectors for subject {}".format(i))
            matfile = self.data_dir / f"subject_{i}.mat"
            # matfile = self.data_dir / f"final_{i}.mat" # name of data
            data = sio.loadmat(str(matfile))
            if i == 1:
                for w_id, curr_word in enumerate(data["words"]):
                    words.append(curr_word[0][0][0][0])
                    times.append(curr_word[0][1][0][0][0])  # words timestamp for nats
                fmri_timestamp = data["time"][0] # fmri data timestamp for nats
                num_tr, num_voxels = data["data"].shape
                reduced_vecs = np.full((num_tr, num_voxels * self.subjects * 2), np.nan)
            vecs = data["data"]
            if pca is not None:
                reduced_vecs[:, (i - 1) * self.vec_dim: i * self.vec_dim] = pca.fit_transform(vecs)
            else:
                subjects_dims.append(subjects_dims[i - 1] + vecs.shape[1])
                reduced_vecs[:, subjects_dims[i - 1]: subjects_dims[i]] = vecs
        reduced_vecs = reduced_vecs[:, ~np.isnan(reduced_vecs).all(axis=0)]
        return words, times, fmri_timestamp, subjects_dims, reduced_vecs

    def hp_dataloader(self):
        self.outfile_dir.mkdir(parents=True, exist_ok=True)
        pca = GaussianRandomProjection(random_state=self.seed, n_components=self.vec_dim) if self.vec_dim else None
        words = []
        times = []
        reduced_vecs = None
        subjects_dims = [0]
        for i in range(1, self.subjects + 1):
            print("Extracting vectors for subject {}".format(i))
            matfile = self.data_dir / f"subject_{i}.mat"
            # matfile = self.data_dir / f"final_{i}.mat" # name of data
            data = sio.loadmat(str(matfile))
            if i == 1:
                for w_id, curr_word in enumerate(data["words"][0]):
                    words.append(curr_word[0][0][0][0])
                    times.append(curr_word[1][0][0])  # words timestamp for HP
                fmri_timestamp = data["time"][:, 0]  # fmri data timestamp for HP
                num_tr, num_voxels = data["data"].shape
                reduced_vecs = np.full((num_tr, num_voxels * self.subjects * 2), np.nan)
            vecs = data["data"]
            if pca is not None:
                reduced_vecs[:, (i - 1) * self.vec_dim: i * self.vec_dim] = pca.fit_transform(vecs)
            else:
                subjects_dims.append(subjects_dims[i - 1] + vecs.shape[1])
                reduced_vecs[:, subjects_dims[i - 1]: subjects_dims[i]] = vecs
        reduced_vecs = reduced_vecs[:, ~np.isnan(reduced_vecs).all(axis=0)]
        return words, times, fmri_timestamp, subjects_dims, reduced_vecs

    def save_files(self, words, words2vecs, subjects_dims):
        filtered_words = []
        index_needed = []
        pure_words = []
        for i, w in enumerate(words):
            w_clean = self.remove_punct(w)[0]
            if w_clean not in {"+", "<punct>"}:
                filtered_words.append(f"{w_clean}_{i}")
                pure_words.append(f"{w_clean}")
                index_needed.append(i)

        pure_words = np.array(pure_words)
        unique_words = list(dict.fromkeys(pure_words))

        for sub in range(1, self.subjects + 1):
            token_level_outfile_name = self.outfile_dir / f"{self.dataset_name}-token-sub--{sub}-{self.lookback}-{self.lookout}-{self.vec_dim}.pth"
            print("Writing vectors to file: " + str(token_level_outfile_name))
            if self.vec_dim:
                vecs_selected = words2vecs[index_needed, (sub - 1) * self.vec_dim: sub * self.vec_dim]
                assert vecs_selected.size() == (len(filtered_words), self.vec_dim)
            else:
                vecs_selected = words2vecs[index_needed, subjects_dims[sub - 1]: subjects_dims[sub]]
                assert vecs_selected.size() == (len(filtered_words), subjects_dims[sub] - subjects_dims[sub - 1])
            torch.save({'dico': filtered_words, 'vectors': vecs_selected.float()}, token_level_outfile_name)

            type_level_outfile_name = self.outfile_dir / f"{self.dataset_name}-type-sub--{sub}-{self.lookback}-{self.lookout}-{self.vec_dim}.pth"
            # vecs = alias_data["vectors"]
            # dico = np.array(alias_data["dico"])
            # if cased == "uncased":
            word_embeddings = torch.empty((len(unique_words), vecs_selected.shape[1]))
            word_indices = [np.where(pure_words == re.sub(r"_\d+", "", w))[0] for w in unique_words]
            for i, indices in enumerate(word_indices):
                word_embeddings[i] = torch.mean(vecs_selected[indices], dim=0)
            # return word_embeddings
            torch.save({'dico': unique_words, 'vectors': word_embeddings.float()}, type_level_outfile_name)

    @staticmethod
    def find_closest(lst, target_num):
        if target_num >= lst[-1]:
            return lst[-1]
        if target_num <= lst[0]:
            return lst[0]
        # find the first index of the number whihc >= target
        pos = bisect.bisect_left(lst, target_num)
        # use the pos, split the list. left: all(val < x for val in lst[lo:i]) ，right: all(val >= x for val in lst[i:hi])
        before = lst[pos - 1]
        after = lst[pos]
        if after - target_num < target_num - before:
            return after
        else:
            return before

    @staticmethod
    def gaussian_smoothing(vs):
        def gaussian_1d(kernel_size=9, sigma=1):
            x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            kernel = np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))
            kernel = kernel / np.sum(kernel)
            return kernel

        weights = gaussian_1d(kernel_size=vs.shape[0])
        # smoothed_vs = np.sum(np.multiply(vs.T, weights),axis=0)
        smoothed_vs = np.dot(weights[:vs.shape[0]], vs)
        return smoothed_vs

    def map_words2vecs(self, words, times, fmri_timestamp, vecs, lookout=4.0, lookback=0.0, delay=6.0, normalize=False):
        """
        Words starts at 20s; fMRI starts at 0s; the delay time is T sec.
        Therefore, the available words starts at (20+lookback)s, and the corresponding fMRI starts at (20+lookback+T)s
        """

        times, inflated_vecs = self.times_vecs_process(times, fmri_timestamp, vecs)
        print(f"inflated vecs shape: {inflated_vecs.shape}")
        m = len(words)
        vecs_out = np.empty((m, inflated_vecs.shape[1]))  # pre-allocate array for input vectors
        for i in range(m):
            # Alignments for words and fmri data (delay).
            start = times.index(self.find_closest(times, times[i] - lookback + delay))
            end = times.index(self.find_closest(times, times[i] + 0.5 + lookout + delay))
            # if start == end:
            #     print("duplicated i", i)
            #     print("start:", start)
            #     print("end:", end)
            # vecs_out[i] = np.mean(vecs_by_timeframe(start, end), axis=1)
            vecs_out[i] = self.gaussian_smoothing(inflated_vecs[start:end])

        if normalize:
            print("=" * 10 + "Do Normalization" + "=" * 10)
            mms = MinMaxScaler()
            vecs_out = mms.fit_transform(vecs_out)

        return vecs_out

    @staticmethod
    def times_vecs_process(times, fmri_timestamp, vecs):
        times = list(np.concatenate((times, np.arange(times[-1] + 0.5, fmri_timestamp[-1] + 0.1, 0.5))))
        gaps_in_times = np.where(np.diff(times) > 2)[0]
        times_starts_index_in_fmri = np.where(fmri_timestamp >= times[0])[0][0]
        new_times = []
        prev_idx = 0
        for idx in gaps_in_times:
            new_times.append(times[prev_idx:idx + 1])
            fill = np.arange(times[idx], times[idx + 1], 0.5)[1:]
            new_times.append(fill)
            prev_idx = idx + 1
        new_times.append(times[prev_idx:])
        times = np.concatenate(new_times)

        repeats = np.diff(np.searchsorted(times, fmri_timestamp, side='right'))
        repeats = np.concatenate(([1], np.where(repeats == 0, 1, repeats)))
        # Inflate vecs using np.repeat
        inflated_vecs = np.repeat(vecs, repeats, axis=0)[times_starts_index_in_fmri:]

        return list(times), inflated_vecs

    @staticmethod
    def inflate(vecs, inflate_size):
        """
        Inflate forward. Repeat every scan 4 times, because one TR has 4 words.
        :param vecs: fmri data, vecs.shape (1351, inflate_size*100)
        :return: inflated_vecs.shape (5404, inflate_size*100)
        """
        inflated = []
        for v in vecs:
            for _ in range(inflate_size):
                inflated.append(v)
        return np.array(inflated)

    @staticmethod
    def remove_punct(word):
        """
        :param word:
        :return: the (possibly modified) word, a boolean variable whether end of sentence
        """
        # word_cleaned = re.findall(r'\b\d{1,2}:\d{2}\s|[ap]\.m\.|Mr\.|Mrs\.|Ms\.|Dr\.|\b\w+-\w+|\b\w+(?:\'\w+)|\w+',word)
        # word_cleaned = re.findall(r'\b\d{1,2}:\d{2}\s|[ap]\.m\.|Mr\.|Mrs\.|Ms\.|Dr\.|\b\w+(?:-\w+)?\b|\b\w+(?:\'\w+)|\w+',word)
        # word_cleaned = re.findall(
        #     r'\b\d{1,2}:\d{2}\s|[ap]\.m\.|Mr\.|Mrs\.|Ms\.|Dr\.|\b\w+(?:\'\w+)|\b(\b\w+(?:-\w+)*|\w+)\b',
        #     word)
        word_cleaned = re.findall(r'\b\d{1,2}:\d{2}\s|[ap]\.m\.|Mr\.|Mrs\.|Ms\.|Dr\.|\b\w+(?:\'\w+)|\b\w+(?:-\w+)*|w+',
                                  word)
        if not word_cleaned:
            # print("MISSING WORD:", word)
            word_cleaned = ["<punct>"]
        return word_cleaned[0], False
