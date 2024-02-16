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

print(x.shape,y.shape)  # (1459, 9) (1459,)

def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series > q1] = np.nan
        print(series.isna().sum())
        data[label] = series
        return data

x = fit_outlier(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=False,random_state=333)

#model
model = Sequential()
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
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
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

# Epoch 1554: early stopping
# loss=1286.423828125
# r2=0.7802103365358142
