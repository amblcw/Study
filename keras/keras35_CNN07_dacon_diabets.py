from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, Input, Conv2D, Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

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
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(456, 6)
# x_test.shape=(196, 6)
# y_train.shape=(456,)
# y_test.shape=(196,)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train).reshape(x_train.shape[0],3,2,1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0],3,2,1)
test_csv = scaler.transform(test_csv).reshape(test_csv.shape[0],3,2,1)

#model
# model = Sequential()
# model.add(Dense(512, input_dim=6, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Conv2D(10,(2,2), padding='same',input_shape=x_train.shape[1:]))
model.add(Conv2D(10, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(1, activation='sigmoid'))

#compile & fit
start_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='max',patience=400,restore_best_weights=True,verbose=1)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/dacon_diabets/K28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1234,batch_size=4,validation_split=0.2,verbose=2,callbacks=[es,mcp])
end_time = time.time()
#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

print(f"Time: {round(end_time-start_time,2)}sec")
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

# MinMaxScaler
# LOSS: 0.5364712476730347
# ACC: 0.7193877696990967

# StandardScaler
# LOSS: 0.5941564440727234
# ACC: 0.7397959232330322

# MaxAbsScaler
# LOSS: 0.5516800284385681
# ACC: 0.7193877696990967

# RobustScaler
# LOSS: 0.58819979429245
# ACC: 0.7448979616165161

# Dropout
# LOSS: 0.5326249599456787
# ACC: 0.7244898080825806

# CPU Time: 118.22sec
# GPU Time: 187.01sec

# CNN 
# Time: 59.62sec
# LOSS: 0.5795493125915527
# ACC: 0.7448979616165161