import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Toolbox import two_d_eq, Assign_features_to_pixels
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math
from itertools import product
import paraHill
from Toolbox import REFINED_Im_Gen
from scipy.spatial.distance import euclidean
import pprint
nn = 100

Euc_Dist = np.zeros(shape=(nn, nn))
for i in range(nn):
    x = pd.read_csv("data/normalized_padel_feats_NCI60_672_small.csv", usecols=[i])
    x = x.iloc[:, 0].to_list()
    for j in range(nn):
        if i <= j:
            continue
        y = pd.read_csv("data/normalized_padel_feats_NCI60_672_small.csv", usecols=[j])
        y = y.iloc[:, 0].to_list()
        Euc_Dist[i][j] = euclidean(x, y)
Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.T)
embedding = MDS(n_components=2)										 # Reduce the dimensionality by MDS into 2 components
mds_xy = embedding.fit_transform(Euc_Dist)					         # Apply MDS
eq_xy = two_d_eq(mds_xy,nn) # -> [0,1]




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
embedding = MDS(n_components=2)										 # Reduce the dimensionality by MDS into 2 components
mds_xy = embedding.fit_transform(transposed_input)					 # Apply MDS
print(np.allclose(transposed_input,transposed_input.T))

eq_xy = two_d_eq(mds_xy,Nn) # -> [0,1]



for i in eq_xy:
    ax[1].scatter(i[0], i[1], color='red')

plt.show()