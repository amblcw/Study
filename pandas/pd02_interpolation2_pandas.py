import pandas as pd
import numpy as np

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
print(data.isna().sum())
# x1    1
# x2    2
# x3    0
# x4    3
print(data.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usage: 288.0 bytes
# None
''' 삭제 '''
print(data.dropna()) # defult가 axis=0
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1))
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

''' 특정값 채우기 '''
print(data.mean())
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
print(data.fillna(data.mean()))
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0
print(data.median())
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
print(data.fillna(data.median()))
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0
print(data.fillna(0))
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0
print(data.fillna(method='ffill')) # 아래와 같음, 권장되지 않음
print(data.fillna(data.ffill()))
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0
print(data.fillna(data.bfill()))
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
data_mean = data
data_mean['x1'] = data['x1'].fillna(data['x1'].mean())
data_mean['x2'] = data['x2'].fillna(data['x1'].ffill())
data_mean['x4'] = data['x4'].fillna(data['x1'].median())
print(data_mean)
#      x1    x2    x3   x4
# 0   2.0   2.0   2.0  6.5
# 1   6.5   4.0   4.0  4.0
# 2   6.0   6.0   6.0  6.5
# 3   8.0   8.0   8.0  8.0
# 4  10.0  10.0  10.0  6.5