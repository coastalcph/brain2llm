import torch
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
# import seaborn as sns
import plotly.express as px
from scipy.spatial import distance

file_X = torch.load('./vectors-mae.pth') 
X = file_X['vectors'].numpy()[:2000]
words_X = file_X['dico'][:2000]

file_Y = torch.load('./vectors-opt.pth')
Y = file_Y['vectors'].numpy()
words_Y = file_Y['dico']

Z = np.concatenate([X, Y])


embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, n_iter=2000).fit_transform(Z)

array_X = embedded[:len(X)]
array_Y = embedded[len(X):]

dict_content = open("./all_pairs.txt").readlines()
images_ids = [i.split(" ",1)[0] for i in dict_content]
images_labels = [i.split(" ",1)[1].strip() for i in dict_content]
# print(dict_fmri)

# selected_words = ['through_3317', 'would_215', 'Ron_1491', 'never_298', 'believe_3239']
selected_words = images_ids
top_n = 5  # Number of closest words to show

distances_YX = distance.cdist(array_Y, array_X, metric='euclidean')
print(distances_YX.shape)

closest_words_Y = {}
for idx, word in enumerate(selected_words):
    if word in words_X:
        word_index_X = words_X.index(word)
        
        closest_word_indices_Y = np.argsort(distances_YX[:, word_index_X])[:top_n]

        candidates = [words_Y[idx] for idx in closest_word_indices_Y]
        if images_labels[idx] in candidates:
            closest_words_Y[word] = candidates

#DF
df = pd.DataFrame(embedded, columns=["X", "Y"])
words = words_X + words_Y
df['words'] = words
df['Modalities'] = ['MAE-Huge'] * len(array_X) + ['OPT_30B'] * len(array_Y)

# Selecting specific words for plotting
selected_words_extended = selected_words + [w for words in closest_words_Y.values() for w in words]
df_filtered = df[df['words'].isin(selected_words_extended)]

selected_words_t = list(closest_words_Y.keys())[:5] + [w for words in list(closest_words_Y.keys())[:5] for w in closest_words_Y[words]]
df_filtered_t = df[df['words'].isin(selected_words_t)]

# df_filtered_t
for word in closest_words_Y:
    print(f"{word}: {', '.join(closest_words_Y[word])}")


