from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

r = int(np.random.uniform(1,1000))
# r = 363
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=r)

#model
model = Sequential()
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
model.fit(x_train,y_train,epochs=500,batch_size=2,verbose=2)

#evalutae & predict
loss = model.evaluate(x_test,y_test)
result = model.predict(x)

print(f"{r=}\n{loss=}\n{result=}")
# r=363
# loss=5.644632339477539

plt.scatter(x,y)
plt.plot(x,result,color='red')
plt.show()

# r=197
# loss=2.023820638656616