from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

# data
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
print(x_data.shape,y_data.shape)    # (4, 2) (4,)

# model
model = LinearSVC(C=10000)
# model = Perceptron()

# fit
model.fit(x_data,y_data)

# score
loss = model.score(x_data,y_data)
y_predict = model.predict(x_data)

acc = accuracy_score(y_data,y_predict)

print("acc",acc)
print("y_data   ",y_data)
print("y_predict",y_predict)

print("=============================")

# model
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(3))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')
es = EarlyStopping(monitor='loss',restore_best_weights=True,patience=100)
model.fit(x_data,y_data,epochs=1000,verbose=2,callbacks=es)

y_predict = np.around(model.predict(x_data).reshape(-1)).astype(int)

acc = accuracy_score(y_data,y_predict)

print("acc",acc)
print("y_data   ",y_data)
print("y_predict",y_predict)