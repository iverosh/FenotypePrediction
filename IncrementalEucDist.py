import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def IncrementalEucDist(n, filename):
    Euc_Dist = np.zeros(shape=(n, n))
    for i in range(n):
        x = pd.read_csv(filename, usecols=[i])
        x = x.iloc[:, 0].to_list()
        for j in range(n):
            if i <= j:
                continue
            y = pd.read_csv(filename, usecols=[j])
            y = y.iloc[:, 0].to_list()
            Euc_Dist[i][j] = euclidean(x, y)
    Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.T)
    return Euc_Dist
