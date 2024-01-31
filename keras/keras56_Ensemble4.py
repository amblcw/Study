import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, LSTM, Input, concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# data
x1_datasets = np.array([range(100),range(301,401)]).T                     # 삼전 종가, 하이닉스 종가
# print(x1_datasets.shape,x2_datasets.shape,x3_datasets.shape)  # (100, 2) (100, 3) (100, 4)

y1 = np.array(range(3001,3101))   # 비트코인 종가
y2 = np.array(range(13001,13101)) # 이더리움 종가

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, y1, y2, train_size=0.7, random_state=333)
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

input1, output1 = make_model1()

merge2 = Dense(10,name='mg2')(output1)
merge3 = Dense(11,name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

merge12 = Dense(10,name='mg12')(output1)
merge13 = Dense(11,name='mg13')(merge12)
last_output2 = Dense(1, name='last2')(merge13)

model = Model(inputs=input1,outputs=[last_output1,last_output2])
model.summary()

# compile
model.compile(loss='mse',optimizer='adam')
model.fit(x1_train,[y1_train,y2_train],epochs=500,verbose=2)

# evaluate
loss = model.evaluate(x1_test,[y1_test,y2_test])
y_predict = model.predict(x1_test)
print(np.array(y_predict).shape)
r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y2_test,y_predict[1])

print(f"loss: {loss}\nr2_1:  {r2_1}\nr2_2:  {r2_2}")

# loss: 9.779135325516108e-06
# r2:  0.9999999834303118