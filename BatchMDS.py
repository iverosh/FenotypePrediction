import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Toolbox import two_d_eq
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math
n = 20 # количество колонок считанных за один раз
readed_n = 0
total_n = 100
mds_xy = np.array([0])
embedding = MDS(n_components=2)										 # Reduce the dimensionality by MDS into 2 components
while readed_n < total_n: # нужно добавить случай если не кратно

    readed_n += n
    Feat_DF = pd.read_csv("data/normalized_padel_feats_NCI60_672_small.csv", usecols=range(readed_n, readed_n+n))     #"data/normalized_padel_feats_NCI60_672_small.csv"
    X = Feat_DF.values                          
    original_input = pd.DataFrame(data = X)

    feature_names_list = Feat_DF.columns.tolist()
    nn = math.ceil(np.sqrt(len(feature_names_list)))      			     # Image dimension
    Nn = original_input.shape[1]                                         # Number of features
    transposed_input = original_input.T 							     # The MDS input data must be transposed , because we want summarize each feature by two values (as compard to regular dimensionality reduction each sample will be described by two values)
    Euc_Dist = euclidean_distances(transposed_input) 					 # Euclidean distance
    Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())   			 # Making the Euclidean distance matrix symmetric
    
    if len(mds_xy) == 1:
        mds_xy = embedding.fit_transform(transposed_input)					 # Apply MDS
    else:
        mds_xy = np.vstack((mds_xy, embedding.fit_transform(transposed_input)))

eq_xy = two_d_eq(mds_xy,total_n) # -> [0,1]

fig, ax = plt.subplots(1, 2)

for i in eq_xy:
    ax[0].scatter(i[0], i[1], color='green')


Feat_DF = pd.read_csv("data/normalized_padel_feats_NCI60_672_small.csv", usecols=range(100))     #"data/normalized_padel_feats_NCI60_672_small.csv"
X = Feat_DF.values                          
original_input = pd.DataFrame(data = X)

feature_names_list = Feat_DF.columns.tolist()
nn = math.ceil(np.sqrt(len(feature_names_list)))      			     # Image dimension
Nn = original_input.shape[1]                                         # Number of features
transposed_input = original_input.T 							     # The MDS input data must be transposed , because we want summarize each feature by two values (as compard to regular dimensionality reduction each sample will be described by two values)
Euc_Dist = euclidean_distances(transposed_input) 					 # Euclidean distance
Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())   			 # Making the Euclidean distance matrix symmetric
mds_xy = embedding.fit_transform(transposed_input)					 # Apply MDS


eq_xy = two_d_eq(mds_xy,Nn) # -> [0,1]



for i in eq_xy:
    ax[1].scatter(i[0], i[1], color='red')

plt.show()