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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from keras.models import load_model
model = load_model("../_data/_save/MCP/keras26_dacon_diabets.hdf5")

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

print(f"LOSS: {loss[0]}\nACC: {loss[1]}\n")

#make submission.CSV
submission_csv['Outcome'] = np.around(y_submit)
import datetime
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour:2}{dt.minute:2}_acc{round(loss[1],4):5}.csv",index=False)


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