from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split

#data
x = np.arange(1,17)
y = np.arange(1,17)

r = int(np.random.uniform(1,1000))
r=404
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=13/16,random_state=r)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=10/13,shuffle=False)

print(x_train,x_val,x_test,sep='\n')
# [10  9 12  3  4 16 14  2  5  7]
# [11 13  8]
# [ 1 15  6]

#model generate
model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(10,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=1,verbose=2,validation_data=(x_val,y_val))

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"{r=}\nLOSS: {loss}\nRESULT: {result}")

# LOSS: 3.031649096259942e-13
# RESULT: [[7.0e+00]
#  [1.1e+07]]

# r=756
# LOSS: 3.78956116703702e-13
# RESULT: [[7.0e+00]
#  [1.1e+07]]

# r=404
# LOSS: 1.5030500435386784e-06
# RESULT: [[7.0000000e+00]
#  [1.0999998e+07]]

# r=837
# LOSS: 1.458981047954e-12
# RESULT: [[6.9999995e+00]
#  [1.0999999e+07]]

# r=413
# LOSS: 7.579122740649855e-14
# RESULT: [[7.0e+00]
#  [1.1e+07]]

# r=333
# LOSS: 0.0
# RESULT: [[7.0000005e+00]
#  [1.1000000e+07]]