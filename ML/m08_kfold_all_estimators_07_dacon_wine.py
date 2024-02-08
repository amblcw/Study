from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0
# print(test_csv)
from m08_addon import m08_classifier
m08_classifier(x,y)

# StandardScaler
# LOSS: 1.0394665002822876
#  ACC:  0.5745454545454546(0.5745454430580139 by loss[1])

# ACC:  [0.44454545 0.44181818 0.44040036 0.43949045 0.43949045]
# 평균 ACC: 0.4411