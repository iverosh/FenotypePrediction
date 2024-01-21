import numpy as np
import pandas as pd
from Toolbox import normalize
df = pd.read_csv('data\phenotype_28042016_synchro.csv', usecols=range(25, 73))

arr = df.values
arr = arr.T
filename = 'memmapped.dat'

mmapped_array = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=arr.shape)
mmapped_array[:] = arr[:]
normalize(mmapped_array)
print(mmapped_array.shape)





