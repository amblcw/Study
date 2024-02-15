import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings(action='ignore')

# data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target 
df_y = pd.DataFrame(y)

# print(df_y)
# print(x,y,x.shape,y.shape,sep='\n')
print(np.unique(y,return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))
zero_num = len(y[np.where(y == 0)]) #y[np.where(조건)]은 조건에 맞는 값들의 인덱스 리스트를 반환
one_num = len(y[np.where(y == 1)])
print(f"0: {zero_num}, 1: {one_num}")
print(df_y.value_counts()) #pandas 요소 개수 세기
print(pd.value_counts(y))  #위와 동일

''' 25퍼 미만 열 삭제 '''
columns = datasets.feature_names
x = pd.DataFrame(x,columns=columns)
print(x.shape)
fi_list = pd.Series([
0,          0.02914825, 0,          0,          0.01078864, 0,
0.03258461, 0.,         0.,         0.,         0.00912187, 0.,
0.00110421, 0.01733133, 0.00827548, 0.00702681, 0.00832807, 0.,
0. ,        0.,         0.69028931, 0.05556586, 0.,         0.01581032,
0.,         0.00624605, 0.,         0.10837919, 0.,         0.,        ])

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print(low_idx_list)

low_col_list = [x.columns[index] for index in low_idx_list]
if len(low_col_list) > len(x.columns) * 0.25:
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print(low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print(x.shape)

from sklearn.decomposition import PCA 
print('x.shape',x.shape)
for i in range(1,x.shape[1]):
    pca = PCA(n_components=i)
    pca_x = pca.fit_transform(x)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(pca_x,y,train_size=0.8,random_state=123)
    param = {'random_state':123}
    model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

    for model in model_list:
        model.fit(x_train,y_train)

        acc = model.score(x_test,y_test)
        print(type(model).__name__,"`s ACC: ",acc,sep='')
        # print(type(model).__name__, ":",model.feature_importances_, "\n")
        
EVR = pca.explained_variance_ratio_
print(EVR)
print(np.cumsum(EVR))

# DecisionTreeClassifier`s ACC: 0.9649122807017544
# DecisionTreeClassifier : [0.         0.02914825 0.         0.         0.01078864 0.
#  0.03258461 0.         0.         0.         0.00912187 0.
#  0.00110421 0.01733133 0.00827548 0.00702681 0.00832807 0.
#  0.         0.         0.69028931 0.05556586 0.         0.01581032
#  0.         0.00624605 0.         0.10837919 0.         0.        ]

# RandomForestClassifier`s ACC: 0.9912280701754386
# RandomForestClassifier : [0.01178004 0.01738715 0.04831914 0.06235906 0.00886921 0.01398074
#  0.05580419 0.12570582 0.0038758  0.00467111 0.01131374 0.00369196
#  0.01043288 0.02893199 0.00503368 0.00531759 0.00436103 0.00710174
#  0.00529289 0.00549803 0.12912118 0.02071569 0.10995985 0.09777345
#  0.01409369 0.01651581 0.03720734 0.11778626 0.00848764 0.00861131]

# GradientBoostingClassifier`s ACC: 0.9736842105263158
# GradientBoostingClassifier : [9.41030000e-04 2.99585337e-02 4.13160665e-04 2.44711991e-03
#  3.71195490e-04 8.43470422e-04 1.50004704e-02 2.98671341e-01
#  0.00000000e+00 3.83097388e-04 5.93179905e-03 2.47930404e-03
#  1.95624989e-03 5.31397408e-03 1.18758355e-03 6.97237950e-04
#  1.95810287e-03 2.55892012e-05 1.52670272e-03 9.95062700e-04
#  4.01934116e-01 5.69421093e-02 1.97152332e-02 3.44981684e-02
#  2.84460874e-03 3.29404951e-04 2.78113151e-02 8.22369071e-02
#  2.02639062e-03 5.60721639e-04]

# XGBClassifier`s ACC: 0.9736842105263158
# XGBClassifier : [0.02699176 0.01672302 0.         0.00923053 0.00389597 0.00380379
#  0.04031224 0.27394462 0.         0.01577737 0.00466982 0.
#  0.00825655 0.00946223 0.01440246 0.0041883  0.         0.00111486
#  0.00345886 0.0039218  0.092041   0.03911273 0.05491628 0.22035155
#  0.00829593 0.00598168 0.02576936 0.10131279 0.00307874 0.00898579]

# EVR
# [9.97535860e-01 2.16969489e-03 1.57691671e-04 1.28259796e-04
#  6.22271482e-06 1.24275059e-06 5.64514181e-07 3.49058549e-07
#  8.72482406e-08 1.09830634e-08 6.21733669e-09 4.80379652e-09
#  2.24633664e-09 1.38645321e-09 8.34012147e-10 5.37532854e-10
#  1.79340515e-10 1.07171269e-10 9.45312917e-11 4.57933794e-11
#  1.20971937e-11 6.52784273e-12]
# [0.99753586 0.99970555 0.99986325 0.99999151 0.99999773 0.99999897
#  0.99999954 0.99999989 0.99999997 0.99999998 0.99999999 0.99999999
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.        ]