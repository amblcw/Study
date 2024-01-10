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

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=False,random_state=333)

#model
model = Sequential()
# model.add(Dense(32,input_dim=9,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(4,activation='relu'))
# model.add(Dense(1,activation='relu'))
model.add(Dense(512,input_dim=9,activation='sigmoid'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))


#compile & fit
model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=1024,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=8192,batch_size=16,validation_split=0.35,verbose=2,callbacks=[es])

#evaluate & predeict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

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

# Epoch 313: early stopping
# 1/5 [=====>........................] - ETA:5/5 [==============================] - 0s 4ms/step
#  1/23 [>.............................] - ET23/23 [==============================] - 0s 524us/step
# loss=2885.126953125
# r2=0.6052954010967706

# Epoch 195: early stopping
# loss=2837.522705078125
# r2=0.6118080026161902

# Epoch 196: early stopping
# loss=2731.427978515625
# r2=0.6263224148597027

# Epoch 271: early stopping
# loss=2706.951416015625
# r2=0.6296709908392599

# Epoch 88: early stopping
# loss=1881.388916015625
# r2=0.6785586008319464

# Epoch 87: early stopping
# loss=1874.8994140625
# r2=0.6796673858225976

# Epoch 455: early stopping <= best
# loss=1431.286376953125
# r2=0.7554601030711634
# model.add(Dense(512,input_dim=9,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1))

# pationce = epo
# loss=1462.5213623046875
# r2=0.7501235084394295

# Epoch 1554: early stopping
# loss=1286.423828125
# r2=0.7802103365358142

# Epoch 2622: early stopping
# loss=1457.3612060546875
# r2=0.75100511942968