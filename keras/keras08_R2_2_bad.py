from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

check = 1
while check >= 0.01 or check < 0:
    r = int(np.random.uniform(1,1000))
    # r=556
    r=100
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=r)

    #model
    model = Sequential()
    model.add(Dense(3,input_dim=1))
    model.add(Dense(10))
    model.add(Dense(30))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(30))
    model.add(Dense(10))
    model.add(Dense(3))
    model.add(Dense(1))

    #compile & fit
    model.compile(loss='mse',optimizer='adam')
    model.fit(x_train,y_train,epochs=100,batch_size=1,verbose=0)

    #evalutae & predict
    loss = model.evaluate(x_test,y_test)
    result = model.predict(x)
    y_prediect = model.predict(x_test)

    r2 = r2_score(y_test,y_prediect)

    print(f"{r=}\n{y_prediect=}\n{loss=}\n{r2=}")
    check = r2
    
# r=100
# y_prediect=array([[14.343742],        
#        [15.779162],
#        [10.037493],
#        [15.061451],
#        [11.472909]], dtype=float32)   
# loss=28.920873641967773
# r2=0.00410207522416195

# plt.scatter(x,y)
# plt.plot(x,result,color='red')
# plt.show()
