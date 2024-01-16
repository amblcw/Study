import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target 
df_y = pd.DataFrame(y)

# print(df_y)
# print(x,y,x.shape,y.shape,sep='\n')
print(np.unique(y,return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))
zero_num = len(y[np.where(y == 0)]) #y[np.where(조건)]은 조건에 맞는 값들의 인덱스 리스트를 반환
one_num = len(y[np.where(y == 1)])
print(f"0: {zero_num}, 1: {one_num}")
print(df_y.value_counts()) #pandas 요소 개수 세기
print(pd.value_counts(y))  #위와 동일

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import load_model
model = load_model("../_data/_save/MCP/keras26_cancer.hdf5")
#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
# accuracy
# print(y_test, np.around(y_predict,0))
def ACC(y_true, y_predict):
    return accuracy_score(y_true, np.around(y_predict))
acc =ACC(y_test,y_predict)
print(acc)
print(f"ACCURACY: {loss[1]}")

# plt.plot(hist.history['accuracy'],color='red',label='accuracy',marker='.')
# plt.plot(hist.history['val_accuracy'],color='blue',label='val_accuracy',marker='.')
# plt.legend(loc='upper right')
# plt.title('california loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# 128~1(step:1/2), all relu, No shuffle
# Epoch 1018: early stopping
# 6/6 [==============================] - 0s 603us/step
# ACCURACY: 0.9766082167625427

# MinMaxScaler
# ACCURACY: 0.9707602262496948

# StandardScaler
# ACCURACY: 0.9707602262496948

# MaxAbsScaler
# ACCURACY: 0.9707602262496948

# RobustScaler
# ACCURACY: 0.9649122953414917