from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Concatenate, Input, Conv2D, Flatten
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)

first = Input(shape=x_train.shape[1:])
c1 = Conv2D(10,(2,2))(first)
print("c1.output.shape")
print(c1.shape)
split1 = Lambda(lambda x: x[:,:,:,:5], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],5))(c1)
split2 = Lambda(lambda x: x[:,:,:,5:], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],5))(c1)
concat = Concatenate()([split1,split2])
flatten = Flatten()(concat)
d1 = Dense(10,activation='relu')(flatten)
output = Dense(1)(d1)
model = Model(inputs=[first],outputs=[output])

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=10, batch_size=1024, verbose=1)

model.summary()

result = model.evaluate(x_test,y_test)

print(result)

