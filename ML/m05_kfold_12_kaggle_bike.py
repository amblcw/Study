from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)

r = int(np.random.uniform(1,1000))
r=2
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

N_SPLIT = 5
kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = RandomForestRegressor()

# fit
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC: ",scores)
print(f"평균 ACC: {round(np.mean(scores),4)}")

""" #### CSV파일 생성 ####
submission_csv['count'] = y_submit
dt = datetime.datetime.now()
# submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}_loss{loss}.csv",index=False)


#### 음수 개수와 RMSLE출력 ####
num_of_minus = submission_csv[submission_csv['count']<0].count()
print("num of minus",num_of_minus['count'])

def RMSLE(y_test,y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

if num_of_minus['count'] == 0:    
    print("RMSLE: ",RMSLE(y_test,y_predict))
else:
    print("음수갯수: ",num_of_minus['count'])
    for i in range(len(y_submit)):
        if y_submit[i] < 0:
            y_submit[i] = 0 """
    
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.title('kaggle bike')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
# plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
# plt.grid()
# plt.legend()
# plt.show()

# r=662
# loss=23495.251953125
# r2=0.2603666422772616
# RMSLE:  1.2877634377136509

# r=662
# loss=22796.43359375
# r2=0.28236574526494185
# RMSLE:  1.3233963468270995

# loss=40385.41015625
# r2=0.15680798684173702
# RMSLE:  1.286986737279555

# loss=43231.234375
# r2=0.09739137590805513
# RMSLE:  1.2574265328392626

# MinMaxScaler
# loss=[38606.78125, 38606.78125]
# r2=0.19394372959029815
# RMSLE:  1.2225483246037947

# StandardScaler
# loss=[40058.3203125, 40058.3203125]
# r2=0.16363740670246008
# RMSLE:  1.2787136000962247

# MaxAbsScaler
# loss=[39243.7109375, 39243.7109375]
# r2=0.1806453152200005
# RMSLE:  1.2434084374789507

# RobustScaler
# loss=[39731.94921875, 39731.94921875]
# r2=0.17045145049558763
# RMSLE:  1.2225427797423805

# LinearSVR
# RMSLE:  1.256727136597206

# ACC list:  [-0.096, 0.0603, 0.0309, -0.1751, 0.0938]
# Best ML:  RandomForestRegressor

# ACC:  [0.27476635 0.29534065 0.28111063 0.32957913 0.31385862]
# 평균 ACC: 0.2989