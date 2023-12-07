import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import matplotlib.pyplot as plt
from Toolbox import two_d_eq
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math

n = 100  # количество фич
m = 19   # количество образцов
mmapped_array = np.memmap("memmapped.dat", mode = "r", shape=(100, 19), dtype='float64')
Euc_Dist = euclidean_distances(mmapped_array)

def my_MDS(dist, m = 2):    
    n = dist.shape[0]
    c = np.eye(n) - np.ones((n, n)) / n
    b = np.dot(c, np.dot(dist, c)) * (-0.5)
    eigenvalues, eigenvectors = np.linalg.eig(b)
    indexes = np.argsort(eigenvalues)[-m:]
    eigenvalues = np.array([eigenvalues[index] for index in indexes])
    eigenvectors = np.array([eigenvectors[index] for index in indexes])
    return np.dot(eigenvectors.T, np.diag(eigenvalues ** 0.5))


mds_xy = my_MDS(Euc_Dist)

eq_xy = two_d_eq(mds_xy, n) # -> [0,1]

fig, ax = plt.subplots(1, 2)

for i in eq_xy:
    ax[0].scatter(i[0], i[1], color='green')


Feat_DF = pd.read_csv("data/normalized_padel_feats_NCI60_672_small.csv", usecols=range(n))     #"data/normalized_padel_feats_NCI60_672_small.csv"
X = Feat_DF.values                          
original_input = pd.DataFrame(data = X)

feature_names_list = Feat_DF.columns.tolist()
nn = math.ceil(np.sqrt(len(feature_names_list)))      			     # Image dimension
Nn = original_input.shape[1]                                         # Number of features
transposed_input = original_input.T 							     # The MDS input data must be transposed , because we want summarize each feature by two values (as compard to regular dimensionality reduction each sample will be described by two values)
Euc_Dist = euclidean_distances(transposed_input) 					 # Euclidean distance
Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())   			 # Making the Euclidean distance matrix symmetric
embedding = MDS(n_components=2)	
mds_xy = embedding.fit_transform(transposed_input)					 # Apply MDS


eq_xy = two_d_eq(mds_xy,Nn) # -> [0,1]



for i in eq_xy:
    ax[1].scatter(i[0], i[1], color='red')

plt.show()
