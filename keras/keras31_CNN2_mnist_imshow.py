import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

print(x_train[0])
print(y_train[0])

import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64)
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892