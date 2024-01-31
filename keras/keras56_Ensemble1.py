import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, LSTM, Input, concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# data
x1_datasets = np.array([range(100),range(301,401)]).T                     # 삼전 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T  # 유가, 환율, 금시세
# print(x1_datasets.shape,x2_datasets.shape)  # (100, 2) (100, 3)

y = np.array(range(3001,3101)) # 비트코인 종가

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.7, random_state=333)
# print(x1_train.shape,x2_train.shape,y_train.shape)  # (70, 2) (70, 3) (70,)

# model 
def make_model1():
    input = Input(shape=(2,))
    d1 = Dense(10, activation='relu')(input)
    d2 = Dense(10, activation='relu')(d1)
    d3 = Dense(10, activation='relu')(d2)
    output = Dense(10, activation='relu')(d3)
    # model = Model(inputs=input, outputs=output)
    # model.summary()
    # return model
    return input, output

def make_model2():
    input = Input(shape=(3,))
    d1 = Dense(100, activation='relu')(input)
    d2 = Dense(100, activation='relu')(d1)
    d3 = Dense(100, activation='relu')(d2)
    output = Dense(5, activation='relu')(d3)
    # model = Model(inputs=input, outputs=output)
    # model.summary()
    # return model
    return input, output

input1, output1 = make_model1()
input2, output2 = make_model2()

merge1 = concatenate([output1,output2], name='mg1')
merge2 = Dense(10,name='mg2')(merge1)
merge3 = Dense(11,name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input2],outputs=last_output)
model.summary()

# compile
model.compile(loss='mse',optimizer='adam')
model.fit([x1_train,x2_train],y_train,epochs=500,verbose=2)

# evaluate
loss = model.evaluate([x1_test,x2_test],y_test)
y_predict = model.predict([x1_test,x2_test])

r2 = r2_score(y_test,y_predict)

print(f"loss: {loss}\nr2:  {r2}")

# loss: 9.779135325516108e-06
# r2:  0.9999999834303118