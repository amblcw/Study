from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

a = np.array([]).reshape(1,1,1,0)
print(a.shape)

#테스트 테스트