from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
'''
IMDB dataset
영화리뷰 데이터셋 
긍정 부정으로 이진분류 
'''

print(x_train[0])                       # 이미 수치화 되어있음
print([len(i) for i in x_train[:10]])   # 패딩 안되어있음
print(np.unique(y_train, return_counts=True)) # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) 이진분류
print(x_train.shape, y_train.shape)     # (25000,) (25000,)
print(x_test.shape, y_test.shape)       # (25000,) (25000,)
print(len(x_train[0]), len(x_test[0]))  # 218 68

''' padding '''
# WORD_LENGTH = max([len(i) for i in x_train]+[len(i) for i in x_test]) # 최대는 터짐.. 
len_list = [len(i) for i in x_train] + [len(i) for i in x_test]
WORD_LENGTH = int(pd.Series(len_list).quantile(q=0.75)) # 단어길이의 3분위수를 기준으로 제한
print("word length: ", WORD_LENGTH) # 285.0

x_train = pad_sequences(x_train, maxlen=WORD_LENGTH)
x_test = pad_sequences(x_test, maxlen=WORD_LENGTH)

''' INPUT_DIM 설정 '''
INPUT_DIM = max([max(i) for i in x_train]+[max(i) for i in x_test]) + 1
print("word kinds: ", INPUT_DIM)    # word kinds:  10000(너무 많아서 1만개로 조정) | 88586(full size)

''' MinMax Scaler '''
""" scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) """

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (25000, 285) (25000,) (25000, 285) (25000,)

# model
model = Sequential()
model.add(Embedding(INPUT_DIM,512))
model.add(Conv1D(128, 3))
model.add(Conv1D(128, 3))
model.add(Conv1D(128, 3))
model.add(MaxPooling1D())
model.add(Conv1D(256, 3))
model.add(Conv1D(256, 3))
model.add(Conv1D(256, 3))
model.add(MaxPooling1D())
model.add(Flatten())
# model.add(LSTM(128, activation='sigmoid'))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))    
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile & fit
s_t = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', patience=20, restore_best_weights=True, verbose=1)
model.fit(x_train,y_train, epochs=2048, validation_data=(x_test,y_test), batch_size=1024, verbose=1, callbacks=[es])
e_t = time.time()

# evaluate
loss = model.evaluate(x_test,y_test)

print(f"time: {e_t-s_t}sec")
print(f"loss: {loss[0]}\nACC:  {loss[1]}")

# time: 605.0370371341705sec
# loss: 0.3250218629837036
# ACC:  0.8664399981498718