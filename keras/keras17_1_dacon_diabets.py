from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#data
path = "C:\\_data\\DACON\\diabets\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0).drop(['Insulin','SkinThickness'],axis=1)
submission_csv = pd.read_csv(path+"sample_submission.csv")

print(train_csv.isna().sum(), test_csv.isna().sum()) # 둘다 결측치 존재하지 않음

test = train_csv['BloodPressure']#.where(train_csv['BloodPressure']==0,train_csv['BloodPressure'].mean())
for i in range(test.size):
    if test[i] == 0:
        test[i] = test.mean()
    # print(test[i])
    
train_csv['BloodPressure'] = test



# print(test)
# zero = train_csv[train_csv['Outcome']==0]
# plt.scatter(range(zero['Insulin'].size),zero['Insulin'],color='red')
# plt.scatter(range(zero['SkinThickness'].size),zero['SkinThickness'],color='blue')
# plt.show()

x = train_csv.drop(['Outcome','Insulin','SkinThickness'],axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False)

#model
model = Sequential()
# model.add(Dense(32, input_dim=8, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(512, input_dim=6, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(800, input_dim = 8, activation='relu'))
# model.add(Dense(600, ))#activation='relu'))
# model.add(Dense(400, ))#activation='relu'))
# model.add(Dense(200, ))#activation='relu'))
# model.add(Dense(128, ))#activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#compile & fit
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='max',patience=400,restore_best_weights=True,verbose=1)
hist = model.fit(x_train,y_train,epochs=2048,batch_size=4,validation_split=0.2,callbacks=[es],verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

print(f"LOSS: {loss[0]}\nACC: {loss[1]}\n")

#make submission.CSV
submission_csv['Outcome'] = np.around(y_submit)
import datetime
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour:2}{dt.minute:2}_acc{round(loss[1],4):5}.csv",index=False)

#plt
plt.plot(hist.history['acc'],color='red',label='accuracy',marker='.')
plt.plot(hist.history['val_acc'],color='blue',label='val_accuracy',marker='.')
plt.legend(loc='upper right')
plt.title('DACON diabets classification')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.grid()
plt.show()

# Epoch 467: early stopping
# LOSS: 0.994541585445404
# ACC: 0.7448979616165161

# Epoch 526: early stopping
# LOSS: 1.219661831855774
# ACC: 0.7653061151504517