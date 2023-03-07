import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time as tm
import csv
import os
import pickle as pk
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter

from .ridge_tools import cross_val_ridge, corr
#import time as tm


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
    return word, eos

def load_transpose_zscore(file): 
    dat = nibabel.load(file).get_data()
    dat = dat.T
    return zscore(dat,axis = 0)

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None,
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

# train/test is the full NLP feature
# train/test_pca is the NLP feature reduced to 10 dimensions via PCA that has been fit on the training data
# feat_dir is the directory where the NLP features are stored
# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def get_nlp_features_fixed_length(layer, seq_len, feat_type, feat_dir, train_indicator, SKIP_WORDS=20, END_WORDS=5176):
    if not os.path.exists(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + 'pca.npy'):
        loaded_origin = np.load(feat_dir + feat_type + '_length_' + str(seq_len) + '_layer_' + str(layer) + '.npy')
        pca = PCA(n_components=100, svd_solver='full', random_state=42) # 501 is over 0.95 for base
        tr_data = pca.fit_transform(loaded_origin)
        np.save(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + 'pca.npy', tr_data)
    loaded = np.load(feat_dir + feat_type + '_length_' + str(seq_len) + '_layer_' + str(layer) + 'pca.npy')
    # loaded = np.load(feat_dir + '/' + feat_type + '_' + str(seq_len)+ '_length'+'_layer_' + str(layer) + '.npy')
    # print(loaded.size())
    print("train_indicator_size:", len(train_indicator))
    # print(np.size(loaded))
    if feat_type == 'elmo':
        train = loaded[SKIP_WORDS:END_WORDS,:][:,:512][train_indicator]   # only forward LSTM
        test = loaded[SKIP_WORDS:END_WORDS,:][:,:512][~train_indicator]   # only forward LSTM
    elif 'bert' in feat_type or feat_type == 'transformer_xl' or feat_type == 'use':
        # print(loaded[SKIP_WORDS:END_WORDS,:].shape())
        print("loaded_size:", len(loaded))
        train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
        test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options elmo, bert, transformer_xl, use'.format(feat_type))
    
    # pca = PCA(n_components=100, svd_solver='full', random_state=42) # 501 is over 0.95 for base
    # pca.fit(train)
    # train_pca = pca.transform(train)
    # test_pca = pca.transform(test)

    return train, test

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def TR_to_word_CV_ind(predict_feat_dict,TR_train_indicator,SKIP_WORDS=20,END_WORDS=5176):
    # time = np.load('./data/fMRI/time_fmri.npy')
    # runs = np.load('./data/fMRI/runs_fmri.npy') 
    # time_words = np.load('./data/fMRI/time_words_fmri.npy')

    time = np.load(predict_feat_dict['time_fmri_path'])
    runs = np.load(predict_feat_dict['runs_fmri_path'])
    time_words = np.load(predict_feat_dict['time_words_path'])

    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)    
    words_id = np.zeros([len(time_words)],dtype=int)
    # w=find what TR each word belongs to
    for i in range(len(time_words)):                
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15 # ??
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True
    return word_train_indicator        


def prepare_fmri_features(predict_feat_dict,train_features, test_features, word_train_indicator, TR_train_indicator, SKIP_WORDS=20, END_WORDS=5176):
        
    # time = np.load('./data/fMRI/time_fmri.npy')
    # runs = np.load('./data/fMRI/runs_fmri.npy') 
    # time_words = np.load('./data/fMRI/time_words_fmri.npy')
    words = np.load('/home/kfb818/projects/b2le/brain_language_nlp/data/stimuli_words.npy')
    time = np.load(predict_feat_dict['time_fmri_path'])
    runs = np.load(predict_feat_dict['runs_fmri_path'])
    time_words = np.load(predict_feat_dict['time_words_path'])

    time_words = time_words[SKIP_WORDS:END_WORDS]
    words = words[SKIP_WORDS:END_WORDS]

        
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
    all_features = np.zeros([time_words.shape[0], train_features.shape[1]])
    all_features[word_train_indicator] = train_features
    all_features[~word_train_indicator] = test_features

    p = all_features.shape[1]
    tmp = np.zeros([time.shape[0], p])
    tr_words = {}
    for i in range(time.shape[0]):
        tmp[i] = np.mean(all_features[(words_id<=i)*(words_id>i-1)],0)
        tr_words[i] = words[(words_id <= i) * (words_id > i - 1)]
    tmp = delay_mat(tmp, np.arange(1,5))
    tr_words = delay_mat(list(tr_words.items()), np.arange(1,5))

    # remove the edges of each run
    tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)])
    tr_words = np.vstack([tr_words[runs == i][20:-15] for i in range(1, 5)])
    tmp = np.nan_to_num(tmp)
        
    return tmp[TR_train_indicator], tmp[~TR_train_indicator], tr_words[TR_train_indicator], tr_words[~TR_train_indicator]
  

def run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict,
                                          regress_feat_names_list = [],method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 4, skip=5):
    
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']
        
        
    n_words = data.shape[0] # number of TRs (4 words in 1 TR)
    n_voxels = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds) # cross-validation marker

    corrs = np.zeros((n_folds, n_voxels)) # voxels
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))

    all_test_Y_features = []
    # all_test_data_p = []
    all_preds = []
    all_test_Y_labels = []

    for ind_num in range(n_folds):
        train_ind = ind!=ind_num # if the index (ind) doesn't equal to current fold number (ind_num), use them as traing part.
        test_ind = ind==ind_num
        word_CV_ind = TR_to_word_CV_ind(predict_feat_dict, train_ind)

        # in original code using pca features
        # _,_,tmp_train_features,tmp_test_features = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)
        # train_features,test_features = prepare_fmri_features(predict_feat_dict, tmp_train_features, tmp_test_features, word_CV_ind, train_ind)

        tmp_train_features, tmp_test_features = get_nlp_features_fixed_length(
            layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)
        train_features, test_features, train_words, test_words= prepare_fmri_features(predict_feat_dict, tmp_train_features, tmp_test_features,
                                                              word_CV_ind, train_ind)

        corrs_lm = np.zeros((n_folds, train_features.shape[1]))

        # skip TRs between train and test data
        if ind_num == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
            # train_words = train_words[skip:, :]
        elif ind_num == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
            # train_words = train_words[:-skip, :]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]
            test_words = test_words[skip:-skip, :]

        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data)))
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data)))
        # all_test_data.append(test_data)

        # PCA brain data
        # train_data_debug = np.random.rand(904,1000)
        pca = PCA(n_components=512, svd_solver='full', random_state=42) # 437 is over 0.95
        pca.fit(train_data)
        train_data_p = pca.transform(train_data)
        # train_data_p = pca.fit_transform(train_data)
        print(sum(pca.explained_variance_ratio_))
        test_data_p = pca.transform(test_data)
        all_test_data_p.append(test_data_p)

        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features))

        all_test_Y_features.append(test_features)
        all_test_Y_labels.append(test_words)

        start_time = tm.time()
        # In original code (predict brain from nlp): train_features are nlp features as X shape(909, 40); train_data are fmri data as Y shape(909, 27905)
        # weights, chosen_lambdas = cross_val_ridge(predict_feat_dict,train_features,train_data, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'plain',do_plot = False)

        # Now (predict nlp from brain): train_features are nlp features which should be Y; train_data are fmri data which should be X (909, 128*4)
        # weights, chosen_lambdas = cross_val_ridge(predict_feat_dict,train_data_p, train_features, n_splits=10,
        #                                           lambdas=np.array([10 ** i for i in range(-6, 10)]), method='plain',
        #                                           do_plot=False)
        weights, chosen_lambdas = cross_val_ridge(predict_feat_dict,train_data, train_features, n_splits=10,
                                                  lambdas=np.array([10 ** i for i in range(-6, 10)]), method='plain',
                                                  do_plot=False)

        # from nlp to fmri
        # preds = np.dot(test_features, weights)
        # corrs[ind_num,:] = corr(preds,test_data)

        # from fmri to nlp
        preds = np.dot(test_data, weights)
        # preds = np.dot(test_data, weights)
        # print(preds.shape)
        corrs_lm[ind_num,:] = corr(preds,test_features)

        all_preds.append(preds)
            
        print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
        del weights

    # return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data)
    return corrs_lm, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_Y_features), np.vstack(all_test_Y_labels)

def binary_classify_neighborhoods(Ypred, Y, n_class=20, nSample = 1000,pair_samples = [],neighborhoods=[]):
    # n_class = how many words to classify at once
    # nSample = how many words to classify

    voxels = Y.shape[-1]
    neighborhoods = np.asarray(neighborhoods, dtype=int)

 #   import time as tm

    acc = np.full([nSample, Y.shape[-1]], np.nan)
    acc2 = np.full([nSample, Y.shape[-1]], np.nan)
    test_word_inds = []

    if len(pair_samples)>0:
        Ypred2 = Ypred[pair_samples>=0]
        Y2 = Y[pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[0]
    start_time = tm.time()
    for idx in range(nSample):
        
        idx_real = np.random.choice(n, n_class)

        sample_real = Y2[idx_real]
        sample_pred_correct = Ypred2[idx_real]

        if len(pair_samples2) == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
            print("something")
            # idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[idx_wrong]

        #print(sample_pred_incorrect.shape)

        # compute distances within neighborhood
        dist_correct = np.sum((sample_real - sample_pred_correct)**2,0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect)**2,0)

        neighborhood_dist_correct = np.array([np.sum(dist_correct[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])
        neighborhood_dist_incorrect = np.array([np.sum(dist_incorrect[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])


        acc[idx,:] = (neighborhood_dist_correct < neighborhood_dist_incorrect)*1.0 + (neighborhood_dist_correct == neighborhood_dist_incorrect)*0.5

        test_word_inds.append(idx_real)
    print('Classification for fold done. Took {} seconds'.format(tm.time()-start_time))
    return np.nanmean(acc,0), np.nanstd(acc,0), acc, np.array(test_word_inds)
