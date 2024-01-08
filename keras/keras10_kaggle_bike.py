from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)
# loss = 99999
# while loss > 23500:
r = int(np.random.uniform(1,1000))
# r=264
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.8,shuffle=False)

print(f"{x_train.shape=},{x_test.shape=}")

# count = 0
# for row in range(len(train_csv)):
#     for colum in range(8):
#         # print(train_csv.iloc[row][colum])
#         if train_csv.iloc[row][colum] < 0:
#             count += 1
# print(count)

#model
model = Sequential()
model.add(Dense(16,input_dim=8,activation='relu'))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dense(128))
# model.add(Dense(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16))#,activation='relu'))
model.add(Dense(8))#,activation='relu'))
model.add(Dense(4))#,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='relu'))
# model.add(Dense(10,input_dim=8))
# model.add(Dense(12))
# model.add(Dense(16))
# model.add(Dense(20))
# model.add(Dense(25))
# model.add(Dense(20))
# model.add(Dense(16))
# model.add(Dense(12))
# model.add(Dense(10))
# model.add(Dense(8))
# model.add(Dense(6))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=64,verbose=2,validation_data=(x_val,y_val))

#evaluate & predict 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\n{loss=}\n{r2=}")
time.sleep(1.5)

# minus_count = 0
# for i in range(len(y_submit)):
#     if y_submit[i] < 0:
#         y_submit[i] = 0
#         minus_count += 1
# print(f"{minus_count}")

#### CSV파일 생성 ####
submission_csv['count'] = y_submit
dt = datetime.datetime.now()
# submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)
submission_csv.to_csv(path+f"submission_{dt.hour}-{dt.minute}_loss{loss}.csv",index=False)


#### 음수 개수와 RMSLE출력 ####
num_of_minus = submission_csv[submission_csv['count']<0].count()
# print(num_of_minus['count'])

def RMSLE(y_test,y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

if num_of_minus['count'] == 0:    
    print("RMSE: ",RMSLE(y_test,y_predict))
else:
    print("음수갯수: ",num_of_minus['count'])
    for i in range(len(y_submit)):
        if y_submit[i] < 0:
            y_submit[i] = 0
    

# r=607
# loss=24400.845703125
# r2=0.27219525073742057

# r=722
# loss=24529.0234375
# r2=0.25317904853894546

# r=487
# loss=22816.34375
# r2=0.2550170339664697

#epo=512, batch=32
# r=129
# loss=24668.87109375
# r2=0.2720856026268641

#epo=1024, batch=64
# r=129
# loss=24819.56640625
# r2=0.26763893229536073

#epo,batch same, layer 8421
# r=129
# loss=24673.1796875
# r2=0.27195846009665237

#epo,batch same, layer peak: 128
# r=129
# loss=24765.69140625
# r2=0.26922854526376827

#epo=2048, batch=64
# r=129
# loss=24635.548828125
# r2=0.273068799946542

#mse -> mae
# r=129
# loss=113.48738861083984
# r2=0.23632385818603485

#레이어 완만하게 증가 감소로 변경
# r=129
# loss=24626.166015625
# r2=0.27334565442093184

#epo=1024, batch=32, 168421 layer
# r=334
# loss=23743.193359375
# r2=0.23967121767106014

# r=292
# loss=24275.77734375
# r2=0.2680544755187566

#relu 
# r=264
# loss=21605.357421875
# r2=0.3421575584596871

# r=686
# loss=23027.189453125
# r2=0.3237820215387406
# RMSE:  1.2671565298278598

# r=315
# loss=21974.43359375
# r2=0.3291111610566304
# RMSE:  1.2724700790915289

# r=77
# loss=21066.880859375
# r2=0.33420764405500225
# RMSE:  1.3063693644350112

#4nd 5nd 에 relu 미적용 <- 현재 제일 점수 좋음
# r=164
# loss=22150.142578125
# r2=0.3058233273560703
# RMSE:  1.2835474664845634

# r=773
# loss=21374.39453125
# r2=0.33442631083109264
# 음수갯수:  18

# r=376
# loss=21489.150390625
# r2=0.34653475537788514
# RMSE:  1.2752383054750283

# r=868
# loss=23561.08203125
# r2=0.2856085488641298
# RMSE:  1.270362516809719