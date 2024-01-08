from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"submission_16-34_loss21774.837890625.csv",index_col=0)  #경로를 적을때 \n같은 경우를 방지하기 위해 \\ 나 /,//도 가능

# print(train_csv)

for i in range(len(train_csv)):
    if train_csv.iloc[i]['count'] < 0:
        train_csv.iloc[i]['count'] = 0
            
print("음수 갯수: ",train_csv[train_csv['count']<0].count())

train_csv.to_csv(path+f"submission_16-34_loss21774.837890625.csv")
