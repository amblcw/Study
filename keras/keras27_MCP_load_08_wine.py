from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

y = to_categorical(y)
print(y,y.shape,sep='\n') #(178,3)

r = int(np.random.uniform(1,1000))
r = 398
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import load_model
model = load_model("../_data/_save/MCP/keras26_wine.hdf5")

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = np.argmax(model.predict(x_test,verbose=0),axis=1)
y_test = np.argmax(y_test,axis=1)

print(f"{r=} \nLOSS: {loss[0]} \nACC:  {accuracy_score(y_test,y_predict)}({loss[1]} by loss[1])")

# r=398
# LOSS: 0.13753800094127655
# ACC:  1.0(1.0 by loss[1])

# MinMaxScaler
# LOSS: 0.11889973282814026
# ACC:  0.9444444444444444(0.9444444179534912 by loss[1])

# StandardScaler
# LOSS: 0.09690777212381363
# ACC:  0.9444444444444444(0.9444444179534912 by loss[1])

# MaxAbsScaler
# LOSS: 0.30003201961517334
# ACC:  0.9722222222222222(0.9722222089767456 by loss[1])

# RobustScaler
# LOSS: 0.15712764859199524
# ACC:  0.9722222222222222(0.9722222089767456 by loss[1])