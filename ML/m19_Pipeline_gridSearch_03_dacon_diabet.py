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

from m19_addon import m19_classifier
m19_classifier(x,y)

# Epoch 526: early stopping
# LOSS: 1.219661831855774
# ACC: 0.7653061151504517

# Best result : SVC`s 0.7194

# ACC:  [0.6870229  0.74045802 0.71538462 0.81538462 0.83076923]
# 평균 ACC: 0.7578
# 평균 ACC: 0.7592

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=7)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 7}
# best score:  0.7799217731421122
# model score:  0.6818181818181818
# ACC:  0.6818181818181818
# y_pred_best`s ACC: 0.6818181818181818
# time: 3.78sec

# Random
# acc :  0.7251908396946565
# time:  2.465505361557007 sec

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 521
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 20
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 60
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 180
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# acc :  0.7175572519083969
# time:  28.79667353630066 sec

# acc :  0.732824427480916
# time:  0.0981907844543457 sec

# acc :  0.7251908396946565
# time:  0.0822453498840332 sec