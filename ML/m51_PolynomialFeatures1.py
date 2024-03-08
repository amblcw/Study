from sklearn.preprocessing import PolynomialFeatures

import pandas as np
import numpy as np

x = np.arange(8).reshape(4,2)
# x = np.arange(24).reshape(8,3)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)
x_pf = x_pf.astype(int)
print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

pf = PolynomialFeatures(degree=2, include_bias=True)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

''' ==================== 컬럼 3개 ==================== '''
x = np.arange(12).reshape(4,3)
pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]
pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)
x_pf = x_pf.astype(int)
print(x_pf)
# [[   0    1    2    0    0    0    1    2    4    0    0    0    0    0
#      0    1    2    4    8]
#  [   3    4    5    9   12   15   16   20   25   27   36   45   48   60
#     75   64   80  100  125]
#  [   6    7    8   36   42   48   49   56   64  216  252  288  294  336
#    384  343  392  448  512]
#  [   9   10   11   81   90   99  100  110  121  729  810  891  900  990
#   1089 1000 1100 1210 1331]]