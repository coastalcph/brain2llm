from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import scipy.io as sio
import os, argparse
import numpy as np
import codecs


def vec2tensor(vector, mapping):
    tensor = np.zeros((51, 61, 23))
    for i in range(len(vector)):
        x, y, z = mapping(i)
        tensor[x][y][z] = vector[i]
    return tensor


def concatenate(matrices):
    """
    Input: n matrices of shape (m, d)
    Output: matrix of shape (m, d*n)
    """
    outmat = []
    m = len(matrices[0]) # 1351
    for i in range(m):
        outmat.append([])
        for mx in matrices: # matrices.shape (n,1351,100); mx.shape(1351,100)
            outmat[i].extend(mx[i])
    return np.array(outmat) # outmat.shape (1351, n*100)


def map_words2vecs(words, times, vecs, normalize=True, lookout=4.0, lookback=0.0, delay=6.0, smoothing='Normal'):
    """
    Words starts at 20s; fMRI starts at 0s; the delay time is T sec.
    Therefore, the available words starts at (20+lookback)s, and the corresponding fMRI starts at (20+lookback+T)s
    """
    def vecavg(vs):
        vsum = np.zeros(vs[0].shape[0])
        for v in vs:
            vsum = vsum + v
        return vsum / len(vs)

    def vecs_by_timeframe(t_start, t_end):
        return vecs[int(t_start*2):int(t_end*2)]  # there are 2 vectors for every second, e.g. vector #40 is at 20 sec

    def gaussian_smoothing(vs):
        def gaussian_1d(kernel_size=9, sigma=1):
            radius = kernel_size // 2
            x = np.arange(-radius, radius + 1)
            phi_x = np.exp(-0.5 / sigma * sigma * x ** 2)
            phi_x = phi_x / phi_x.sum()
            return phi_x[::-1]

        weights = gaussian_1d(kernel_size=vs.shape[0])
        # smoothed_vs = np.sum(np.multiply(vs.T, weights),axis=0)
        smoothed_vs = np.dot(vs.T, weights[:vs.shape[0]])
        # print(len(weights[:vs.shape[0]]))
        return smoothed_vs.T

    vecs_out = []
    m = len(words)
    for i in range(m):
        # Alignments for words and fmri data (delay).
        start = times[i] - lookback + delay
        end = times[i] + 0.5 + lookout +delay
        if smoothing != 'Gaussian':
        # vecs_out.append(vecavg(vecs_by_timeframe(start, end)))
            vecs_out.append(np.mean(vecs_by_timeframe(start, end), axis=1))
        else:
            vecs_out.append(gaussian_smoothing(vecs_by_timeframe(start, end)))

    if normalize:
        mms = MinMaxScaler()
        vecs_out = mms.fit_transform(vecs_out)
    words2vecs = [(w, vs) for w, vs in zip(words, vecs_out)]
    return words2vecs


def inflate(vecs):
    """
    Inflate forward. Repeat every TR 4 times, because one TR has 4 words.
    :param vecs: fmri data, vecs.shape (1351, n*100)
    :return: inflated_vecs.shape (5404, n*100)
    """
    inflated = []
    for v in vecs:
        for _ in range(4):
            inflated.append(v)
    return np.array(inflated)


def remove_punct(word):
    """
    :param word:
    :return: the (possibly modified) word, a boolean variable whether end of sentence
    """
    word = ''.join(ch for ch in word if ch not in "‘\"")
    exceptions = ["Mr.", "Mrs.", "Ms.", "Dr."]
    if word in exceptions:
        return word, False
    if word == '--':
        return "<punct>", True
    eos = False
    eospunct = set(".?!…—")
    for p in eospunct:
        if word.endswith(p):
            eos = True
        word = word.strip(p)
    eospunct.update(set("@,:;\""))
    word = ''.join(ch for ch in word if ch not in eospunct)
    # prevent empty word
    if word == "":
        word = "<punct>"
    else:
        if word != "+" and not word[-1].isalpha():
            word = word[:-1]
        else:
            pass
    return word, eos


def convert(data_dir, outfile_dir, dataset_name, vec_dim=100, subjects=8, normalize=True, lookout=4.0, lookback=0.0, delay=6.0, smoothing="Normal"):
    if vec_dim:
        pca = PCA(n_components=vec_dim, random_state=42)
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)
    words = []
    times = []
    reduced_vecs = []
    subjects_dims = [0]
    for i in range(1, subjects+1):
        print("Extracting vectors for subject {}".format(i))
        matfile = os.path.join(data_dir, "subject_{}.mat".format(i))
        data = sio.loadmat(matfile)
        if i == 1:
            for w in data["words"][0]:
                words.append(w[0][0][0][0])
                times.append(w[1][0][0]) # words timestamp
            fmri_timestamp = data["time"][:,0] # fmri data timestamp
        # vec2tensor_map = data["meta"][0][0][6]
        vecs = data["data"]
        if vec_dim:
            reduced_vecs.append(pca.fit_transform(vecs))  # 1351 TRs, to be mapped to 5176 words
        else:
            reduced_vecs.append(vecs)
            subjects_dims.append(subjects_dims[i-1] + vecs.shape[1])

    # reduced_vecs = concatenate(np.array(reduced_vecs))
    reduced_vecs = np.hstack(reduced_vecs)
    reduced_vecs = inflate(reduced_vecs)
    print("inflated vecs shape: {}".format(reduced_vecs.shape))

    # get the feature based on all the features in a window
    words2vecs = map_words2vecs(words, times, reduced_vecs, normalize, lookout, lookback, delay, smoothing=smoothing)

    # fileidx = 0
    for sub in range(1, subjects + 1):
        # wc = 0
        # sc = 0
        outfile_name = "{}/{}-sub--{}-{}-{}-{}.txt".format(outfile_dir, dataset_name , sub, lookback, lookout, vec_dim)
        print("Writing vectors to file: " + outfile_name)
        outfile = codecs.open(outfile_name, "w", "utf-8")
        outfile.write(f"4898 {vec_dim}\n" if vec_dim else f"4898 {subjects_dims[sub]}\n")
        for idx, (w, v) in enumerate(words2vecs):
            # eos = False
            w, eos = remove_punct(w)
            if w == "+" or w == "<punct>":
                # of.write("\n")
                continue
            outfile.write("{}_{} ".format(w.lower(), idx))
            # wc += 1
            # for i in range(v.shape[0]):
            #     outfile.write("\t{}".format(v[i]))
            if vec_dim:
                outfile.write(' '.join([str(v) for v in v.tolist()[(sub - 1) * vec_dim: sub * vec_dim]]))
            else:
                outfile.write(' '.join([str(v) for v in v.tolist()[subjects_dims[sub-1] : subjects_dims[sub]]]))
            outfile.write("\n")
            # if eos:
            #     sc += 1
            #     outfile.write("\n")
                # if sc in [41, 82]:  # make splits here
                    # fileidx += 1
        outfile.close()


def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # data_dir = '../fmri/data'
    data_dir = '/home/kfb818/projects/datasets/fmri-sohmm/fmri/data'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default=data_dir, help='Features and labels')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--output', '-o', help="Output file", default='potter')
    parser.add_argument('--dim', type=int, help="Dimensionality of features per subject", default=128)
    parser.add_argument('--subjects', '-s', type=int, help="Number of subjects", default=8)
    parser.add_argument('--normalize', dest='normalize', help="Column-wise normalization of features", action='store_true')
    parser.add_argument('--no-normalize', '-N', dest='normalize', help="No column-wise normalization of features", action='store_false')
    parser.add_argument('--lookout', type=float, help="future context", default=2.0)
    parser.add_argument('--lookback', type=float, help="past context", default=2.0)
    parser.add_argument('--delay', type=float, help="delay t seconds", default=6.0)
    parser.add_argument('--smoothing', type=str, help="smoothing methods [Normal, Gaussian]", default='Gaussian')
    parser.set_defaults(normalize=True)
    args = parser.parse_args()
    convert(args.data, args.output, args.dim, args.subjects, args.normalize, args.lookout, args.lookback, args.delay, args.smoothing)

if __name__ == "__main__":
    main()