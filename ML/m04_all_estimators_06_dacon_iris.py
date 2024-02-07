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

r = int(np.random.uniform(1,1000))
# r=326
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

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

# r=326
# LOSS: 0.3992086946964264
# ACC:  1.0(1.0by loss[1])