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
r = int(np.random.uniform(1,1000))
# r = 894
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#model
from sklearn.utils import all_estimators

all_algorithms = all_estimators(type_filter='classifier')
# all_algorithms = all_estimators(type_filter='regressor')
# print(len(all_algorithms))  # 41(분류) 55(회귀) 
result_list = []
error_list = []
for name, algorithm in all_algorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        acc = model.score(x_test,y_test)
    except Exception as e:
        print(f"{name:30} ERROR")
        error_list.append(e)
        continue
    print(f"{name:30} ACC: {acc:.4f}")
    result_list.append((name,acc))
    
# print('error_list: \n',error_list)
best_result = max(result_list)[1]
best_algirithm = result_list[result_list.index(max(result_list))][0]
print(f'\nBest result : {best_algirithm}`s {best_result:.4f}')

# StandardScaler
# LOSS: 1.0394665002822876
#  ACC:  0.5745454545454546(0.5745454430580139 by loss[1])