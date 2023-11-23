import numpy as np
import pandas as pd

# читаем CSV файл в DataFrame
df = pd.read_csv('data\\normalized_padel_feats_NCI60_672_small.csv', usecols=range(100))

# преобразуем DataFrame в ndarray
arr = df.values
arr = arr.T
# создаем новый memmap массив и записываем в него данные из ndarray
filename = 'memmapped.dat'
print(arr.shape)
mmapped_array = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=arr.shape)
mmapped_array[:] = arr[:]
print(arr)





