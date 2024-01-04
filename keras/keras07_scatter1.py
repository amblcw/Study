from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#data
x = np.array(range(1,11))   
y = np.array([1,2,3,4,6,5,7,8,9,10])

r = int(np.random.uniform(0,1000))
print(r)

r=218

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=r) #test + train > 1 => error, test + train < 1 => data손실

print(f"{x_train=}",f"{y_train=}",f"{x_test=}",f"{y_test=}",sep='\n')

#model generate
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict(x)
print(f"random_state = {r}")
print(f"LOSS: {loss}\nRESULT: {result}")

plt.scatter(x,y)
plt.plot(x, result, color='red')
plt.show()

# random_state = 218
# LOSS: 0.11576584726572037
# RESULT: [[11.5002575]]