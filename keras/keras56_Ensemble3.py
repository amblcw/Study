import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, LSTM, Input, concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# data
x1_datasets = np.array([range(100),range(301,401)]).T                     # 삼전 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T  # 유가, 환율, 금시세
x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
print(x1_datasets.shape,x2_datasets.shape,x3_datasets.shape)  # (100, 2) (100, 3) (100, 4)

y1 = np.array(range(3001,3101))   # 비트코인 종가
y2 = np.array(range(13001,13101)) # 이더리움 종가

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, x2_datasets, x3_datasets,
                                                                                               y1, y2, train_size=0.7, random_state=333)
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

def make_model3():
    input = Input(shape=(4,))
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
input3, output3 = make_model3()

merge1 = concatenate([output1,output2,output3], name='mg1')
merge2 = Dense(10,name='mg2')(merge1)
merge3 = Dense(11,name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

merge12 = Dense(10,name='mg12')(merge1)
merge13 = Dense(11,name='mg13')(merge2)
last_output2 = Dense(1, name='last2')(merge3)

model1 = Model(inputs=[input1,input2,input3],outputs=[last_output1,last_output2])
model1.summary()

# compile
model1.compile(loss='mse',optimizer='adam')
model1.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs=500,verbose=2)

# evaluate
loss1 = model1.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
y1_predict = model1.predict([x1_test,x2_test,x3_test])

r21 = r2_score(y1_test,y1_predict[0])
r22 = r2_score(y2_test,y1_predict[1])

print(f"model 1 \nloss: {loss1}\nr21:  {r21}\nr22:  {r22}")

# loss: [1.9240567684173584, 0.6828665137290955, 1.2411903142929077]
# r21:  0.998842956379219
# r22:  0.9978969370960357