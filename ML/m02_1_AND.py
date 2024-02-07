from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# data
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,0,0,1])
print(x_data.shape,y_data.shape)    # (4, 2) (4,)

# model
# model = LinearSVC(C=100)
model = Perceptron()

# fit
model.fit(x_data,y_data)

# score
loss = model.score(x_data,y_data)
y_predict = model.predict(x_data)

acc = accuracy_score(y_data,y_predict)

print("acc",acc)
print("y_data   ",y_data)
print("y_predict",y_predict)