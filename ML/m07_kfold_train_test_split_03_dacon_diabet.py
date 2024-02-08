from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')
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

from m07_addon import m07_classifier
m07_classifier(x,y)

# Epoch 526: early stopping
# LOSS: 1.219661831855774
# ACC: 0.7653061151504517

# Best result : SVC`s 0.7194

# ACC:  [0.6870229  0.74045802 0.71538462 0.81538462 0.83076923]
# 평균 ACC: 0.7578
# 평균 ACC: 0.7592