from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import math
import datetime


path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  #경로를 적을때 \n같은 경우를 방지하기 위해 \\ 나 /,//도 가능

# print(train_csv)
# train_csv = train_csv.fillna(-1)

# print(train_csv.shape)

nan_index = np.where(np.isnan(train_csv))

row, colum = nan_index

for i in range(len(row)):
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    pre = train_csv.iloc[row[i]-1,colum[i]]
    next = train_csv.iloc[row[i]+1,colum[i]]
    train_csv.iloc[row[i],colum[i]] = (pre + next)/2
    print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    

#     for row in train_csv.iloc[:,n]:
#         if row == :
#             print("nan")        
        
#         p_pre = pre
#         pre = row
    # print("next")