from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from acc import ACC

# data
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
print(x_data.shape,y_data.shape)    # (4, 2) (4,)

# model
model = Sequential()
model.add(Dense(1,input_dim=2,activation='sigmoid'))

# fit
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_data,y_data,epochs=100)

# score
loss = model.evaluate(x_data,y_data)
y_predict = np.around(model.predict(x_data).reshape(-1)).astype(int)

acc = accuracy_score(y_data,y_predict)

print("acc",ACC)
print("y_data   ",y_data)
print("y_predict",y_predict)
