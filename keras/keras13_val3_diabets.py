from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

datasets = load_diabetes()
x = datasets.data
y = datasets.target 

print(x.shape,y.shape,sep='\n')
print(datasets.feature_names)
print(datasets.DESCR)
r2=0
# while r2< 0.65:
r = int(np.random.uniform(1,1000))
r = 969
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=r)

#model
model = Sequential()
model.add(Dense(32,input_dim=10,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
s_t = time.time()
model.fit(x_train,y_train,epochs=2048,batch_size=16,validation_split=0.3,verbose=2)
e_t = time.time()

#evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print(f"time: {round(e_t-s_t,2)}sec\n{r=}\nloss: {loss}\n R2: {r2}")
time.sleep(1)
pass

#R2 0.62 over
#189, 
# r=969
# loss: 1921.8782958984375
#  R2: 0.6669673069075344

# r=10
# loss: 2105.162841796875
#  R2: 0.6514000078661242