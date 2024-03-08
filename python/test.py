from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping

#data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)

#model
model = Sequential()
model.add(Dense(16,input_dim=10,activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam',metrics='f1_score')
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=16,validation_split=0.3,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=2)
y_predict = model.predict(x_test,verbose=0)
r2 = r2_score(y_test,y_predict)

print(f"{loss=}\n{r2=}")

# Epoch 142: early stopping
# 2/2 - 0s - loss: 2985.1555 - 16ms/epoch - 8ms/step
# 2/2 [==============================] - 0s 1ms/step
# loss=2985.155517578125
# r2=0.4861588420968208

# Epoch 144: early stopping
# 2/2 - 0s - loss: 2990.7566 - 17ms/epoch - 8ms/step
# loss=2990.756591796875
# r2=0.48519479541792887