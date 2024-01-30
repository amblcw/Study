from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
# from function_package import split_x, split_xy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "C:\_data\KAGGLE\Jena_Climate_Dataset\\"

datasets = pd.read_csv(path+"jena_climate_2009_2016.csv", index_col=0)

print(datasets.columns)
col = datasets.columns

'''MinMaxScale'''
minmax = MinMaxScaler()
datasets = minmax.fit_transform(datasets)
datasets = pd.DataFrame(datasets,columns=col)   # 다시 DataFrame으로, 이유는 밑의 함수들을 이용하기 위해서

print(type(datasets))

row_x = datasets
row_y = datasets['T (degC)']

print(row_x.shape,row_y.shape)

# print(row_x.isna().sum(),row_y.isna().sum())    #결측치 존재하지 않음

'''data RNN에 맞게 변환'''
def split_xy(data, time_step, y_col):
    result_x = []
    result_y = []
    
    num = len(data) - time_step
    for i in range(num):
        result_x.append(data[i : i+time_step])
        result_y.append(data.iloc[i+time_step][y_col])
    
    return np.array(result_x), np.array(result_y)

x, y = split_xy(datasets,3,'T (degC)')

print(x.shape,y.shape)      #(420548, 3, 14) (420548,)
print(x[0],y[0],sep='\n')   #검증완료

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=333)

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

# model
model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# compile & fit
model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=4096,batch_size=8192,validation_split=0.2,verbose=2,callbacks=[es])

# evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)

print(f"LOSS: {loss}\nR2:  {r2}")
    

# LOSS: 1.190172042697668e-05