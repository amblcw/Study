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

from keras.layers import Input, Dense, Conv2D, BatchNormalization, Concatenate
from keras.activations import swish
def conv_block(filters:int,input_layer,kernel:tuple,padding:str='valid',strides:int=1):
    conv = Conv2D(filters,kernel,padding=padding,strides=strides)(input_layer)
    batchNorm = BatchNormalization()(conv)
    output_layer = swish(batchNorm)
    return output_layer, filters

def bottleneck(filters,input_layer,shortcut:bool):
    c1, _ = conv_block(filters,input_layer,kernel=(3,3),padding='same',strides=1)
    c2, _ = conv_block(filters,c1,kernel=(3,3),padding='same',strides=1)
    output_layer = c2
    if shortcut:
        output_layer = c2 + input_layer
    return output_layer

def c2f(filters:int,input_layer,shorcut:bool,n:int):
    c1, c1_out = conv_block(filters,input_layer,kernel=(1,1))
    split_num = int(c1_out/2)
    split1 = Lambda(lambda x: x[:,:,:,:split_num], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],split_num))(c1)
    split2 = Lambda(lambda x: x[:,:,:,split_num:], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],c1.shape[3]-split_num))(c1)
    pre_layer = split2
    bottleneck_output_list = [split1,split2]
    for i in range(n):
        output = bottleneck(split_num,pre_layer,shortcut=shorcut)
        bottleneck_output_list.append(output)
    concat = Concatenate()(bottleneck_output_list)
    output_layer, _ = conv_block(filters,concat,(1,1))
    return output_layer



(x_train,y_train), (x_test,y_test) = cifar10.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)

first = Input(shape=x_train.shape[1:])
# c1, c1_out = conv_block(32,first,kernel=(3,3),padding='same')
# print("c1.output.shape")
# print(c1.shape)
# split_num = int(c1_out/2)
# split1 = Lambda(lambda x: x[:,:,:,:split_num], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],split_num))(c1)
# split2 = Lambda(lambda x: x[:,:,:,split_num:], output_shape=(c1.shape[0],c1.shape[1],c1.shape[2],c1.shape[3]-split_num))(c1)
# concat = Concatenate()([split1,split2])
# print(f"{concat.shape=}")
# b1 = bottleneck(64,concat,True)
b1 = c2f(64,first,shorcut=True,n=2)
b1 = c2f(128,b1,shorcut=True,n=4)

flatten = Flatten()(b1)
d1 = Dense(10,activation='relu')(flatten)
output = Dense(1)(d1)
model = Model(inputs=[first],outputs=[output])
model.summary()

# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train, epochs=10, batch_size=1024, verbose=1)


# result = model.evaluate(x_test,y_test)

# print(result)

