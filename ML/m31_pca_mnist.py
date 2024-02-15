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

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=x.shape[1])
x1 = pca.fit_transform(x)
EVR = pca.explained_variance_ratio_
EVR_sum = np.cumsum(EVR)
evr_sum = pd.Series(EVR_sum).round(decimals=6)
print(evr_sum)
print(len(evr_sum[evr_sum >= 0.95]))
print(len(evr_sum[evr_sum >= 0.99]))
print(len(evr_sum[evr_sum >= 0.999]))
print(len(evr_sum[evr_sum >= 1.0]))
print("0.95  커트라인 n_components: ",len(evr_sum[evr_sum < 0.95]))
print("0.99  커트라인 n_components: ",len(evr_sum[evr_sum < 0.99]))
print("0.999 커트라인 n_components: ",len(evr_sum[evr_sum < 0.999]))
print("1.0   커트라인 n_components: ",len(evr_sum[evr_sum < 1.0]))
print(evr_sum.iloc[331])    # 0.950031
print(evr_sum.iloc[543])    # 0.990077
print(evr_sum.iloc[682])    # 0.999023
print(evr_sum.iloc[712])    # 1.0

cutline = [
    (len(evr_sum[evr_sum < 0.95]), round(evr_sum.iloc[331],4)),
    (len(evr_sum[evr_sum < 0.99]), round(evr_sum.iloc[543],4)),
    (len(evr_sum[evr_sum < 0.999]), round(evr_sum.iloc[682],4)),
    (len(evr_sum[evr_sum < 1.0]), round(evr_sum.iloc[712],4)),
    (784, '전체 데이터')
]



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
for cut_idx, cut_num in cutline:
    pca = PCA(n_components=cut_idx)
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
    print(f"{cut_num}의 ACC:  {loss[1]}\n\n") 
    acc_list.append((round(end_time - start_time,4), cut_num,round(loss[1],4)))
    
for time, c_n , acc in acc_list:
    print(f"{c_n:<6}의 ACC:  {acc}") 
    print(f"time: {time}sec")
    
""" 
0.95의 ACC:  0.9731
time: 25.5278sec
0.9901의 ACC:  0.9689
time: 25.5964sec
0.999의 ACC:  0.9696
time: 26.2743sec
1.0의 ACC:  0.9685
time: 26.3754sec
전체 데이터의 ACC:  0.9681
time: 25.9777sec
"""