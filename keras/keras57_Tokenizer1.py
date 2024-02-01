from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import function_package as fp

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# 많이 나온것 우선, 같은 개수면 먼저 나온 것 우선
print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

TIME_STEP = 3

x = token.texts_to_sequences([text])
print(x)
# [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]
x = np.array(x).reshape(-1,1)
x1 = to_categorical(x)
x1 = x1[:,1:]
print(x1.shape) # (12, 8)
# sklearn
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(x)
print(x2.shape) # (12, 8)
# pandas 
x3 = pd.get_dummies(x.reshape(-1), dtype=int)
print(x3.shape) # (12, 8) 

x1_rnn = fp.split_x(x1,TIME_STEP)
x2_rnn = fp.split_x(x2,TIME_STEP)
x3_rnn = fp.split_x(x3,TIME_STEP)
print(x1_rnn.shape,x2_rnn.shape,x3_rnn.shape)   # (10, 3, 8) (10, 3, 8) (10, 3, 8)