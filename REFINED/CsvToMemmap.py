import numpy as np
import pandas as pd
from Toolbox import normalize
df = pd.read_csv('REFINED\\data\\total_df_for_aio_chickpea_28042016_synchro.csv', usecols=range(1, 401))# nrows=400)
for x in df.columns:
    print(f"'{x}'", end = ", ")

arr = df.values
arr = arr.T
filename = 'memmapped.dat'

mmapped_array = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=arr.shape)
mmapped_array[:] = arr[:]
normalize(mmapped_array)
print(mmapped_array.shape)
print(mmapped_array.dtype)
# for x in mmapped_array:
#     print(*x)

    





