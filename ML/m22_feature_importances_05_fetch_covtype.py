from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')


datasets = fetch_covtype()
x = datasets.data  
y = datasets.target

x = x[:100000]
y = y[:100000]
y -= 1
# print(x.shape,y.shape)      #(581012, 54) (581012,)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y.reshape(-1,1))

# print(y,y.shape,sep='\n')
# print(np.count_nonzero(y[:,0]))
'''
sklearn : (581012, 7)
pandas  : (581012, 7)
keras   : (581012, 8)
keras 첫번째 열이 미심직어 찍어보니
print(np.count_nonzero(y[:,0])) # 0
따라서 첫번째 열 잘라내고 슬라이싱
'''
# print(y.shape)

# print(y,y.shape,sep='\n')       # (581012, 7)
# print(np.count_nonzero(y[:,0])) # 211840
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
# r=994
# LOSS: 0.1615818589925766
# ACC:  0.9583371580686616(0.9583371877670288 by loss[1])

# ACC list:  [0.7122, 0.5692, 0.7246, 0.9243, 0.9332, 0.9526]
# Best ML:  RandomForestClassifier

# ACC:  [0.95458809 0.9541492  0.95613673 0.95528476 0.95555154]
# 평균 ACC: 0.9551

# 최적의 매개변수:  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터:  {'min_samples_split': 2, 'n_jobs': 2}
# best score:  0.9430888888888889
# model score:  0.9469
# ACC:  0.9469
# y_pred_best`s ACC: 0.9469
# time: 296.03sec

# Random
# acc :  0.9256
# time:  34.10847043991089 sec

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 2962
# max_resources_: 80000
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 2962
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 8886
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 26658
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 79974
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc :  0.9454
# time:  82.62311005592346 sec

# acc :  0.9462
# time:  5.9261651039123535 sec

""" DecisionTreeClassifier`s ACC: 0.93245
DecisionTreeClassifier : [3.06620586e-01 2.13547348e-02 1.18632561e-02 5.80914662e-02
 3.93649743e-02 1.66656614e-01 2.83840216e-02 2.97563862e-02
 1.85690284e-02 1.39119785e-01 1.21942994e-01 2.23657093e-04
 1.62545596e-03 8.61765761e-04 9.11371461e-05 6.15082056e-04
 2.11905060e-03 3.60738977e-03 3.31499321e-04 5.70613556e-04
 0.00000000e+00 1.75909781e-04 0.00000000e+00 2.54432555e-03
 6.57507872e-04 5.45445939e-03 7.60962656e-04 1.14784068e-04
 0.00000000e+00 5.13912928e-03 1.17699732e-03 0.00000000e+00
 8.55022187e-04 2.49904846e-03 0.00000000e+00 7.69328817e-04
 6.75350135e-03 1.28574382e-03 0.00000000e+00 0.00000000e+00
 1.26832440e-04 4.59921103e-05 8.25825697e-03 4.34573131e-03
 4.34678225e-04 1.32020465e-03 3.40799807e-04 1.01797075e-04
 7.95112408e-04 4.89719208e-05 0.00000000e+00 1.88902986e-03
 2.03961946e-03 2.96755225e-04]

RandomForestClassifier`s ACC: 0.9429
RandomForestClassifier : [2.39279592e-01 4.50191435e-02 3.09084445e-02 5.96687261e-02
 5.83198957e-02 1.28367839e-01 3.74528291e-02 4.28381744e-02
 3.99246659e-02 1.11187535e-01 5.68881687e-02 1.78567328e-03
 1.95445604e-02 2.70265095e-02 5.09017281e-04 1.85134374e-03
 5.00984115e-03 4.10343974e-03 2.67073959e-04 1.19695250e-03
 8.48594588e-05 1.30631450e-04 2.24079811e-06 8.48246520e-03
 1.02317386e-03 1.83320664e-02 1.76325540e-03 2.15096591e-04
 0.00000000e+00 2.96245939e-03 1.98850402e-03 8.04901725e-04
 1.08756535e-03 3.26669897e-03 6.97826594e-05 1.48672068e-03
 1.01778545e-02 1.87879600e-03 5.54496807e-06 1.08661730e-04
 4.26110052e-05 2.64287984e-05 8.80873888e-03 6.09123333e-03
 6.83681374e-04 1.47237601e-03 1.01869157e-03 7.46691269e-05
 6.21855639e-04 2.73377654e-05 3.58585990e-04 6.44281408e-03
 5.54486011e-03 3.76541147e-03]

GradientBoostingClassifier`s ACC: 0.87005
GradientBoostingClassifier : [4.46842335e-01 1.38249521e-02 9.27712685e-04 2.92568562e-02
 2.44928354e-02 1.23747845e-01 1.05562913e-02 1.41471436e-02
 6.28549704e-03 7.66222227e-02 1.53146669e-01 3.16958049e-04
 9.93362669e-03 2.87529990e-03 5.29233424e-04 2.02555098e-03
 9.36328676e-03 4.56934971e-03 3.48696866e-04 1.07938832e-03
 5.02625012e-05 3.20391290e-05 0.00000000e+00 1.63049638e-02
 6.13580823e-04 1.76237332e-02 2.02402255e-03 2.00982153e-05
 0.00000000e+00 4.74047505e-03 8.28128172e-04 3.93862481e-04
 0.00000000e+00 1.36680499e-03 7.13315031e-06 3.63092778e-04
 8.87596604e-03 2.21868961e-03 7.15811331e-06 1.05431572e-04
 7.04759531e-05 9.59406397e-05 1.64066023e-03 7.29775233e-03
 4.62955810e-04 6.96162034e-04 6.28847674e-04 6.87266976e-05
 3.25595675e-04 8.50824712e-05 1.14701808e-05 1.15808897e-03
 8.94536347e-04 9.65115017e-05]

XGBClassifier`s ACC: 0.93915
XGBClassifier : [0.05731834 0.00484608 0.00290477 0.01049566 0.0082596  0.01700958
 0.0073812  0.00660743 0.00359998 0.01327327 0.33082515 0.00452776
 0.04745989 0.08252537 0.00320749 0.00571807 0.02867439 0.0117828
 0.00452883 0.00424985 0.00392159 0.0039458  0.         0.03269626
 0.00462651 0.06881102 0.00362759 0.0010354  0.         0.02140854
 0.0069805  0.02599352 0.00746446 0.00945428 0.00157603 0.00675081
 0.03316964 0.01040383 0.         0.00202824 0.00464344 0.00202734
 0.01091633 0.02956955 0.0034728  0.00754324 0.0046618  0.00158798
 0.00784722 0.00094331 0.00047106 0.0104285  0.01400207 0.00279584] """