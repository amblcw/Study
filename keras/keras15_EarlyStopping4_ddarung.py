from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#data
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)

#model
model = Sequential()
model.add(Dense(32,input_dim=9,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='relu'))

#compile & fit
model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=40)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=32,validation_split=0.3,verbose=2,callbacks=[es])

#evaluate & predeict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

import datetime
dt = datetime.datetime.now()
submission_csv['count'] = y_submit
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)

r2 = r2_score(y_test,y_predict)
print(f"{loss=}\n{r2=}")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
plt.legend()
plt.show()

# Epoch 161: early stopping
# 5/5 [==============================] - 0s 0s/step
# 23/23 [==============================] - 0s 761us/step
# loss=2830.26513671875
# r2=0.6128008278870951

# Epoch 166: early stopping
# 5/5 [==============================] - 0s 0s/step
# 23/23 [==============================] - 0s 746us/step
# loss=2882.05810546875
# r2=0.6057152176580647