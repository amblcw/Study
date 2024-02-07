from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

# while r2 < 0.8:
r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)


# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train),np.max(x_train))
# print(np.min(x_test),np.max(x_test))

#model
from sklearn.utils import all_estimators

# all_algorithms = all_estimators(type_filter='classifier')
all_algorithms = all_estimators(type_filter='regressor')
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
# loss=[4.916057586669922, 1.7392468452453613]
# r2=0.9333945146886241
# RMSE: 2.217218335403849