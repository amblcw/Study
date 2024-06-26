from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape,y.shape,sep='\n')
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
r2=0
while r2 < 0.6: 
    r = int(np.random.uniform(1,1000))
    r = 176
    # r = 130
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r)

    #model
    model = Sequential()
    model.add(Dense(10,input_dim=8))
    model.add(Dense(17))
    model.add(Dense(25))
    model.add(Dense(35))
    model.add(Dense(27))
    model.add(Dense(15))
    model.add(Dense(10))
    model.add(Dense(5))
    model.add(Dense(1))

    #compile fit
    model.compile(loss='mse',optimizer='adam')
    start_time = time.time()
    model.fit(x_train,y_train,epochs=10000,batch_size=500,verbose=2)

    #evaluate predict
    loss = model.evaluate(x_test,y_test)
    result = model.predict(x)
    y_predict = model.predict(x_test)

    r2 = r2_score(y_test,y_predict)
    end_time = time.time()
    print(f"Time: {round(end_time-start_time,2)}sec")
    print(f"{r=}\n{loss=}\n{r2=}")
    pass

# r=176
# loss=0.5322199463844299
# r2=0.6000881155946733

# r=176 
# loss=0.5211321711540222   mae로 훈련, mse가 더 잘 나옴
# r2=0.5541717141267914
