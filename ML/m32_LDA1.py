# 스케일링 후 LDA로 교육용 파일들 만들기
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# x = np.append(x_train,x_test, axis=0)
# x = np.concatenate([x_train,x_test], axis=0)
x = np.vstack([x_train,x_test])
print(x.shape)  # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.append(y_train,y_test)

scaler = StandardScaler()
x = scaler.fit_transform(x)

'''
ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
LDA의 n_components는 x의 [feature개수]와 [y라벨 종류-1] 둘 보다 작게 설정해야한다
'''
lda = LinearDiscriminantAnalysis(n_components=9).fit(x,y)
x1 = lda.transform(x)
EVR = lda.explained_variance_ratio_
EVR_sum = np.cumsum(EVR)
evr_sum = pd.Series(EVR_sum).round(decimals=6)
print(evr_sum)


import matplotlib.pyplot as plt
plt.plot(evr_sum)
plt.grid()
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

pca = PCA(n_components=x_train.shape[1]-1).fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

acc_list = []
# for cut_idx, cut_num in cutline:
# pca = PCA(n_components=cut_idx)
pca_x = pca.fit_transform(x)

x_train = pca_x[:60000]
x_test = pca_x[60000:]
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# model
model = Sequential()
model.add(Dense(200, input_shape=x_train.shape[1:],activation='relu'))          
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(units=10, activation='softmax'))

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', patience=256, restore_best_weights=True)
hist = model.fit(x_train, y_train, batch_size=256, epochs=256, validation_split=0.2, verbose=2 )
end_time = time.time()
# evaluate & predict
loss = model.evaluate(x_test,y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)

print(f"time: {end_time - start_time}sec")
print("ACC", loss[1])
# print(f"{cut_num}의 ACC:  {loss[1]}\n\n") 
# acc_list.append((round(end_time - start_time,4), cut_num,round(loss[1],4)))
    
for time, c_n , acc in acc_list:
    print(f"{c_n:<6}의 ACC:  {acc}") 
    print(f"time: {time}sec")
    
""" 
0.95  의 ACC:  0.9747
time: 63.7506sec
0.9901의 ACC:  0.9716
time: 64.8889sec
0.999 의 ACC:  0.9696
time: 65.8921sec
1.0   의 ACC:  0.9674
time: 66.8954sec
전체 데이터의 ACC:  0.9701
time: 66.0671sec
"""

# time: 69.01050925254822sec
# ACC 0.9690999984741211