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

#data
path = "C:\\_data\\DACON\\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

print(x,y,sep='\n')
print(x.shape,y.shape)  #x:(120, 4) y:(120,) test_csv:(30, 4)

y = y.to_frame('species')                                      #pandas Series에서 dataframe으로 바꾸는 법


# print(x)
print(f"{x.shape=}, {y.shape=}")      
# print(np.unique(y, return_counts=True)) 
from m07_addon import m07_classifier
m07_classifier(x,y)
# r=326
# LOSS: 0.3992086946964264
# ACC:  1.0(1.0by loss[1])

# ACC:  [0.95833333 0.95833333 0.95833333 0.875      0.95833333]
# 평균 ACC: 0.9417