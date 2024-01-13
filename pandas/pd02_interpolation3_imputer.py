import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]
                     ])

data = data.T
data.columns = ['x1','x2','x3','x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

s_imputer = SimpleImputer()
data2 = s_imputer.fit_transform(data)
print(data2)    # 평균값으로 채워짐
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]
s_imputer = SimpleImputer(strategy='mean')
data2 = s_imputer.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]
s_imputer = SimpleImputer(strategy='median')
data2 = s_imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  6.]
#  [ 7.  4.  4.  4.]
#  [ 6.  4.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  6.]]
s_imputer = SimpleImputer(strategy='most_frequent')
data2 = s_imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  2.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]]
s_imputer = SimpleImputer(strategy='constant')  # 0 들어감
data2 = s_imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  0.]
#  [ 0.  4.  4.  4.]
#  [ 6.  0.  6.  0.]
#  [ 8.  8.  8.  8.]
#  [10.  0. 10.  0.]]
s_imputer = SimpleImputer(strategy='constant', fill_value=777)  # 777 들어감
data2 = s_imputer.fit_transform(data)
print(data2)
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]]
knn_imputer = KNNImputer()  # 가까운데 어떤게 더 많이 잇냐로 보간
data3 = knn_imputer.fit_transform(data)
print(data3)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]
iter_imputer = IterativeImputer()   # interpolater와 비슷하다, 선형회귀
data4 = iter_imputer.fit_transform(data)
# print(data4)
# [[ 2.          2.          2.          2.0000005 ]
#  [ 4.00000099  4.          4.          4.        ]
#  [ 6.          5.99999928  6.          5.9999996 ]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99999872 10.          9.99999874]]

from impyute import mice    # linear_regressor, logistic_regressor, randomforest등 5가지의 알고리즘으로 예측해서 보간한다
# from impyute.imputation.cs import mice   
print(np.__version__)   # 1.22.4
aaa = mice(data.values)
print(aaa)
# [[ 2.          2.          2.          1.99944722]
#  [ 4.00110565  4.          4.          4.        ]
#  [ 6.          5.99956991  6.          5.99967235]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99913982 10.          9.99989748]]