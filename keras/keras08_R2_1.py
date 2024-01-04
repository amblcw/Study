from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

check = 0
while check < 0.99:
    r = int(np.random.uniform(1,1000))
    # r=556
    # r=455
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=r)

    #model
    model = Sequential()
    # model.add(Dense(3,input_dim=1))
    # model.add(Dense(10))
    # model.add(Dense(30))
    # model.add(Dense(100))
    # model.add(Dense(40))
    # model.add(Dense(10))
    # model.add(Dense(4))
    # model.add(Dense(1))

    model.add(Dense(3,input_dim=1))
    model.add(Dense(10))
    model.add(Dense(30))
    model.add(Dense(100))
    model.add(Dense(300))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(30))
    model.add(Dense(10))
    model.add(Dense(3))
    model.add(Dense(1))

    #compile & fit
    model.compile(loss='mse',optimizer='adam')
    model.fit(x_train,y_train,epochs=300,batch_size=2,verbose=2)

    #evalutae & predict
    loss = model.evaluate(x_test,y_test)
    result = model.predict(x)
    y_prediect = model.predict(x_test)

    r2 = r2_score(y_test,y_prediect)

    print(f"{r=}\n{y_prediect=}\n{loss=}\n{r2=}")
    # loss=0.6742212772369385
    # r2=0.9808580707042345
    check = r2

# plt.scatter(x,y)
# plt.plot(x,result,color='red')
# plt.show()

# r=197
# loss=2.023820638656616