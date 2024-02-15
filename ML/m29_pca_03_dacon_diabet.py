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

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
fi_list = pd.Series([0.0412366,  0.35054714, 0.09804412, 0.17476757, 0.19195972, 0.14344486])

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

low_col_list = [x.columns[index] for index in low_idx_list]
if len(low_col_list) > len(x.columns) * 0.25:
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

from sklearn.decomposition import PCA 
origin_x = x
print('x.shape',x.shape)
for i in range(1,x.shape[1]):
    pca = PCA(n_components=i)
    x = pca.fit_transform(origin_x)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
    param = {'random_state':123}
    model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

    for model in model_list:
        model.fit(x_train,y_train)

        acc = model.score(x_test,y_test)
        print(type(model).__name__,"`s ACC: ",acc,sep='')
        print(type(model).__name__, ":",model.feature_importances_, "\n")
        
EVR = pca.explained_variance_ratio_
print(EVR)
print(np.cumsum(EVR))

# EVR 
# [0.76905854 0.12314332 0.07296987 0.03475029]
# [0.76905854 0.89220186 0.96517173 0.99992201]

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

# DecisionTreeClassifier`s ACC: 0.6793893129770993
# DecisionTreeClassifier : [0.0412366  0.35054714 0.09804412 0.17476757 0.19195972 0.14344486]

# RandomForestClassifier`s ACC: 0.732824427480916
# RandomForestClassifier : [0.0918769  0.29576223 0.09566886 0.19343958 0.16693426 0.15631816]

# GradientBoostingClassifier`s ACC: 0.6946564885496184
# GradientBoostingClassifier : [0.06366833 0.41473097 0.03642243 0.2240571  0.13785934 0.12326184]

# XGBClassifier`s ACC: 0.7099236641221374
# XGBClassifier : [0.13485736 0.29204473 0.10412243 0.17463374 0.12552099 0.16882078]